import argparse
import itertools
import functools
import logging
from copy import deepcopy
from pathlib import Path

import healpy as hp
import galsim.roman
import metadetect_driver
import numpy as np
from pyimcom.analysis import OutImage
from pyimcom.config import Settings

import astropy
import matplotlib as mpl
import matplotlib.pyplot as plt
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
from astropy import units as u
from astropy.coordinates import (search_around_sky, SkyCoord)
from scipy import stats


logging.basicConfig(level=logging.INFO)


# pc.field("is_primary")

flagged = ~(
    (pc.field("gauss_flags") == 0)
    & (pc.field("pgauss_flags") == 0)
    & (pc.field("psfrec_flags") == 0)
    & (pc.field("gauss_T_flags") == 0)
    & (pc.list_element(pc.field("pgauss_band_flux_flags"), 0) == 0)
    & (pc.list_element(pc.field("pgauss_band_flux_flags"), 1) == 0)
    & (pc.list_element(pc.field("pgauss_band_flux_flags"), 2) == 0)
    # & (pc.field("gauss_s2n") > 5)
    # & (pc.field("pgauss_T_ratio") > 0.5)
    # & (pc.field("mfrac") < 1e-3)
)

ROMAN_BAND_KEYS = {
    "W": "W146",
    "R": "R062",
    "Z": "Z087",
    "Y": "Y106",
    "J": "J129",
    "H": "H158",
    "F": "F184",
    "K": "K213",
}

ROMAN_BANDPASSES = galsim.roman.getBandpasses()

BANDS = ["Y", "J", "H"]


def _main(input_dir, output_dir, truth_dir, mosaic, save=False, show=False):

    report_path = Path("reports/")
    report_path.mkdir(parents=True, exist_ok=True)

    input_image = Path(input_dir) / f"H{mosaic}_coadds"/ f"im3x2-H{mosaic}_00_00.cpr.fits.gz"
    outimage = OutImage(input_image)

    block_wcs = metadetect_driver.get_imcom_wcs(outimage)
    mosaic_nside = outimage.cfg.Nside * outimage.cfg.nblock
    mosaic_wcs = astropy.wcs.WCS(header={"NAXIS1": mosaic_nside, "NAXIS2": mosaic_nside}, naxis=2)
    mosaic_wcs.wcs.crpix = [
        (mosaic_nside + 1) / 2.0,
        (mosaic_nside + 1) / 2.0,
    ]
    mosaic_wcs.wcs.cdelt = [-outimage.cfg.dtheta, outimage.cfg.dtheta]
    mosaic_wcs.wcs.ctype = ["RA---STG", "DEC--STG"]
    mosaic_wcs.wcs.crval = [outimage.cfg.ra, outimage.cfg.dec]
    mosaic_wcs.wcs.lonpole = outimage.cfg.lonpole

    # wcs_footprint = wcs.calc_footprint()
    mosaic_wcs_footprint = mosaic_wcs.calc_footprint()

    # OpenUniverse2024 catalogs in NSIDE 32 and RING ordering
    NSIDE = 32
    RING = True
    NEST = not RING
    LONLAT = True
    covering_healpixels = hp.query_polygon(
        NSIDE,
        hp.ang2vec(mosaic_wcs_footprint[:, 0], mosaic_wcs_footprint[:, 1], lonlat=LONLAT),
        inclusive=True,
        nest=NEST,
        fact=max(4, 4096 // NSIDE),
    )
    covering_healpixels = np.unique(covering_healpixels)
    print(f"HEALPix: {covering_healpixels}")

    # TODO check if the file exists -- possibly not fully covered
    galaxy_dataset_paths = [
        Path(truth_dir) / f"galaxy_{pix}.parquet"
        for pix in covering_healpixels
    ]
    galaxy_flux_dataset_paths = [
        Path(truth_dir) / f"galaxy_flux_{pix}.parquet"
        for pix in covering_healpixels
    ]

    pointsource_dataset_paths = [
        Path(truth_dir) / f"pointsource_{pix}.parquet"
        for pix in covering_healpixels
    ]
    pointsource_flux_dataset_paths = [
        Path(truth_dir) / f"pointsource_flux_{pix}.parquet"
        for pix in covering_healpixels
    ]

    _galaxy_dataset = ds.dataset(galaxy_dataset_paths)
    _galaxy_flux_dataset = ds.dataset(galaxy_flux_dataset_paths)
    galaxy_dataset = _galaxy_dataset.join(_galaxy_flux_dataset, "galaxy_id")
    galaxy_table = galaxy_dataset.to_table()

    _pointsource_dataset = ds.dataset(pointsource_dataset_paths)
    # NB pyarrow joins don't support StructType, so excise that first
    _pointsource_table = ds.dataset(pointsource_dataset_paths).to_table(columns=[s.name for s in _pointsource_dataset.schema if not isinstance(s.type, pa.StructType)])
    _pointsource_dataset = ds.dataset(_pointsource_table)
    _pointsource_flux_dataset = ds.dataset(pointsource_flux_dataset_paths)
    pointsource_dataset = _pointsource_dataset.join(_pointsource_flux_dataset, "id")
    pointsource_table = pointsource_dataset.to_table()

    galaxy_coords = SkyCoord(galaxy_table["ra"], galaxy_table["dec"], unit="deg")
    pointsource_coords = SkyCoord(pointsource_table["ra"], pointsource_table["dec"], unit="deg")

    galaxy_in_footprint = mosaic_wcs.footprint_contains(galaxy_coords)
    pointsource_in_footprint = mosaic_wcs.footprint_contains(pointsource_coords)

    galaxy_coords = galaxy_coords[galaxy_in_footprint]
    pointsource_coords = pointsource_coords[pointsource_in_footprint]

    galaxy_table = galaxy_table.filter(galaxy_in_footprint)
    pointsource_table = pointsource_table.filter(pointsource_in_footprint)

    # ---

    outpath = Path(output_dir) / f"YJH{mosaic}_catalogs" / "noshear"

    print(f"Flags: {flagged}")
    detection_dataset = ds.dataset(outpath)
    detection_table = detection_dataset.to_table(filter=pc.field("is_primary"))

    detection_coords = SkyCoord(detection_table["ra"], detection_table["dec"], unit="deg")

    match_distance = Settings.pixscale_native / Settings.arcsec * 5  # 5 pixels (pixscale is 0.11 asec)

    # galaxy_detection_index, galaxy_detection_distance, _ = galaxy_coords.match_to_catalog_sky(detection_coords)
    # pointsource_detection_index, pointsource_detection_distance, _ = pointsource_coords.match_to_catalog_sky(detection_coords)
    detection_galaxy_match = search_around_sky(detection_coords, galaxy_coords, match_distance * u.arcsec)
    detection_pointsource_match = search_around_sky(detection_coords, pointsource_coords, match_distance * u.arcsec)

    detection_indices = np.indices(detection_coords.shape).reshape(-1)

    # not_pointsource = np.setdiff1d(detection_indices, detection_pointsource_match.indices_to_first_set)
    # not_galaxy = np.setdiff1d(detection_indices, detection_galaxy_match.indices_to_first_set)

    _detection_galaxy_unique_indices, _detection_galaxy_unique_counts = np.unique(detection_galaxy_match.indices_to_first_set, return_counts=True)
    detection_galaxy_unique_indices = np.extract(_detection_galaxy_unique_counts == 1, _detection_galaxy_unique_indices)
    detection_galaxy_ambiguous_indices = np.extract(_detection_galaxy_unique_counts != 1, _detection_galaxy_unique_indices)

    _detection_pointsource_unique_indices, _detection_pointsource_unique_counts = np.unique(detection_pointsource_match.indices_to_first_set, return_counts=True)
    detection_pointsource_unique_indices = np.extract(_detection_pointsource_unique_counts == 1, _detection_pointsource_unique_indices)
    detection_pointsource_ambiguous_indices = np.extract(_detection_pointsource_unique_counts != 1, _detection_pointsource_unique_indices)

    # star or galaxy
    # (Pdb) np.intersect1d(detection_galaxy_match.indices_to_first_set, detection_pointsource_match.indices_to_first_set)
    # star or galaxy but unique in each -- this is the tricky case that we need to handle extra
    # (Pdb) np.intersect1d(detection_galaxy_unique_indices, detection_pointsource_unique_indices)
    detection_ambiguous_indices = np.intersect1d(detection_galaxy_unique_indices, detection_pointsource_unique_indices)

    unique_galaxy = np.setdiff1d(detection_galaxy_unique_indices, detection_pointsource_match.indices_to_first_set)
    unique_pointsource = np.setdiff1d(detection_pointsource_unique_indices, detection_galaxy_match.indices_to_first_set)
    ambiguous = functools.reduce(np.union1d, [detection_galaxy_ambiguous_indices, detection_pointsource_ambiguous_indices, detection_ambiguous_indices])

    unmatched = np.setdiff1d(detection_indices, np.union1d(detection_galaxy_match.indices_to_first_set, detection_pointsource_match.indices_to_first_set))

    assert len(np.intersect1d(unique_galaxy, unique_pointsource)) == 0
    assert len(np.intersect1d(unique_galaxy, ambiguous)) == 0
    assert len(np.intersect1d(unique_galaxy, unmatched)) == 0
    assert len(np.intersect1d(unique_pointsource, ambiguous)) == 0
    assert len(np.intersect1d(unique_pointsource, unmatched)) == 0
    assert len(np.intersect1d(ambiguous, unmatched)) == 0
    assert len(unique_galaxy) + len(unique_pointsource) + len(ambiguous) + len(unmatched) == len(detection_indices)
    # assert sum(no_match & unique_match) == 0
    # assert sum(no_match & ambiguous_match) == 0
    # assert sum(unique_match & ambiguous_match) == 0
    # assert sum(no_match | unique_match | ambiguous_match) == len(coords)

    print(f"total: {len(detection_indices)}")
    print(f"no match: {len(unmatched)}")
    print(f"unique galaxy match: {len(unique_galaxy)}")
    print(f"unique pointsource match: {len(unique_pointsource)}")
    print(f"ambiguous match: {len(ambiguous)}")

    matched_galaxies = galaxy_table.take(detection_galaxy_match.indices_to_second_set)
    matched_detections = detection_table.take(detection_galaxy_match.indices_to_first_set)
    # detected_stars = detection_table.take(detection_pointsource_match.indices_to_first_set)
    # TODO how can we apply a quality filter after the fact...?

    fig, axs = plt.subplots(1, 1)

    image = outimage.get_coadded_layer("SCI")
    _max = np.max(np.abs(image))
    _vmin = -_max
    _vmax = +_max
    norm = mpl.colors.AsinhNorm(1e-3, vmin=_vmin, vmax=_vmax)
    cmap = mpl.cm.twilight_shifted

    # detection_xs, detection_ys = block_wcs.world_to_pixel(detection_coords)  # FIXME why does this not match up...?
    _in_block = (
        (pc.list_element(pc.field("block_id"), 0) == 0)
        & (pc.list_element(pc.field("block_id"), 1) == 0)
    )
    detection_flagged_xs = detection_table.filter(flagged & _in_block)["sx_col"].to_numpy()
    detection_flagged_ys = detection_table.filter(flagged & _in_block)["sx_row"].to_numpy()
    detection_unflagged_xs = detection_table.filter(~flagged & _in_block)["sx_col"].to_numpy()
    detection_unflagged_ys = detection_table.filter(~flagged & _in_block)["sx_row"].to_numpy()
    galaxy_xs, galaxy_ys = block_wcs.world_to_pixel(galaxy_coords)
    pointsource_xs, pointsource_ys = block_wcs.world_to_pixel(pointsource_coords)

    nrow, ncol = image.shape

    axs.imshow(image, origin="lower", norm=norm, cmap=cmap)

    axs.scatter(
        galaxy_xs[(galaxy_xs > 0) & (galaxy_ys > 0) & (galaxy_xs < ncol) & (galaxy_ys < nrow)],
        galaxy_ys[(galaxy_xs > 0) & (galaxy_ys > 0) & (galaxy_xs < ncol) & (galaxy_ys < nrow)],
        ec='k',
        fc='none',
        marker='o',
        s=72,
        ls=':',
        label='True Galaxy',
    )
    axs.scatter(
        pointsource_xs[(pointsource_xs > 0) & (pointsource_ys > 0) & (pointsource_xs < ncol) & (pointsource_ys < nrow)],
        pointsource_ys[(pointsource_xs > 0) & (pointsource_ys > 0) & (pointsource_xs < ncol) & (pointsource_ys < nrow)],
        ec='k',
        fc='none',
        marker='s',
        s=72,
        ls=':',
        label='True Star',
    )
    axs.scatter(
        detection_flagged_xs[(detection_flagged_xs > 0) & (detection_flagged_ys > 0) & (detection_flagged_xs < ncol) & (detection_flagged_ys < nrow)],
        detection_flagged_ys[(detection_flagged_xs > 0) & (detection_flagged_ys > 0) & (detection_flagged_xs < ncol) & (detection_flagged_ys < nrow)],
        c='k',
        marker='x',
        s=72,
        label='Flagged',
    )
    axs.scatter(
        detection_unflagged_xs[(detection_unflagged_xs > 0) & (detection_unflagged_ys > 0) & (detection_unflagged_xs < ncol) & (detection_unflagged_ys < nrow)],
        detection_unflagged_ys[(detection_unflagged_xs > 0) & (detection_unflagged_ys > 0) & (detection_unflagged_xs < ncol) & (detection_unflagged_ys < nrow)],
        c='k',
        marker='+',
        s=72,
        label='Unflagged',
    )
    axs.set_xlim(0, ncol)
    axs.set_ylim(0, nrow)
    axs.set_aspect(1)

    axs.legend(loc="upper right")

    # fig.colorbar(mpl.cm.ScalarMappable(norm, cmap), ax=axs, label='Flux [$e^- / (0.11 arcsec)^2 / exposure$]', ticks=ticker)

    if save:
        fig.savefig(report_path / "report-image.png")
        fig.savefig(report_path / "report-image.pdf")
    if show:
        plt.show()
    plt.close()

    # ---

    print(f"Found {detection_table.num_rows} detections")

    fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)

    norm = mpl.colors.LogNorm()

    bins=[
        np.geomspace(0.5, 10_000, 101),
        np.linspace(-0.5, 2, 101),
    ]

    axs[1].hist2d(
        detection_table.filter(~flagged)["gauss_s2n"],
        detection_table.filter(~flagged)["gauss_T_ratio"],
        bins=bins,
        norm=norm,
        rasterized=True,
    )
    axs[0].hist2d(
        detection_table.filter(flagged)["gauss_s2n"],
        detection_table.filter(flagged)["gauss_T_ratio"],
        bins=bins,
        norm=norm,
        rasterized=True,
    )

    axs[0].set_title("Flagged")
    axs[1].set_title("Unflagged")
    for ax in axs:
        ax.set_xscale("log")
        ax.set_xlabel("$gauss\\_s2n$")
        ax.set_ylabel("$gauss\\_T\\_ratio$")

    if save:
        fig.savefig(report_path / "report-size_snr.png")
        fig.savefig(report_path / "report-size_snr.pdf")
    if show:
        plt.show()
    plt.close()

    # ---

    fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)

    norm = mpl.colors.LogNorm()

    bins=[
        np.linspace(-2.0, 2.0, 101),
        np.linspace(-2.0, 2.0, 101),
    ]

    axs[1].hist2d(
        -2.5 * np.log10(
            pc.divide(
                pc.list_element(detection_table.filter(~flagged)["pgauss_band_flux"], 0),
                pc.list_element(detection_table.filter(~flagged)["pgauss_band_flux"], 1),
            )
        ),
        -2.5 * np.log10(
            pc.divide(
                pc.list_element(detection_table.filter(~flagged)["pgauss_band_flux"], 1),
                pc.list_element(detection_table.filter(~flagged)["pgauss_band_flux"], 2),
            )
        ),
        bins=bins,
        norm=norm,
        rasterized=True,
    )
    axs[0].hist2d(
        -2.5 * np.log10(
            pc.divide(
                pc.list_element(detection_table.filter(flagged)["pgauss_band_flux"], 0),
                pc.list_element(detection_table.filter(flagged)["pgauss_band_flux"], 1),
            )
        ),
        -2.5 * np.log10(
            pc.divide(
                pc.list_element(detection_table.filter(flagged)["pgauss_band_flux"], 1),
                pc.list_element(detection_table.filter(flagged)["pgauss_band_flux"], 2),
            )
        ),
        bins=bins,
        norm=norm,
        rasterized=True,
    )

    axs[0].set_title("Flagged")
    axs[1].set_title("Unflagged")
    for ax in axs:
        ax.set_xlabel("$Y - J$")
        ax.set_ylabel("$J - H$")

    if save:
        fig.savefig(report_path / "report-color_color.png")
        fig.savefig(report_path / "report-color_color.pdf")
    if show:
        plt.show()
    plt.close()

    # ---

    fig, axs = plt.subplots(1, len(BANDS), sharex=True, sharey=True)

    bins=[
        np.linspace(15, 30, 101),
        np.linspace(15, 30, 101),
    ]

    for i, band in enumerate(BANDS):

        axs[i].hist2d(
            -2.5 * np.log10(matched_galaxies[f"roman_flux_{ROMAN_BAND_KEYS[band]}"]) + ROMAN_BANDPASSES[ROMAN_BAND_KEYS[band]].zeropoint,
            -2.5 * np.log10(pc.list_element(matched_detections["pgauss_band_flux"], i)) + ROMAN_BANDPASSES[ROMAN_BAND_KEYS[band]].zeropoint,
            norm="log",
            bins=bins,
            rasterized=True,
        )

        axs[i].axline((20, 20), (21, 21), ls=":")

        axs[i].set_aspect(1)
        axs[i].set_title(band)

    fig.supxlabel("True [mag]")
    fig.supylabel("Measured [pgauss]")

    if save:
        plt.savefig(report_path / f"report-color.png")
        plt.savefig(report_path / f"report-color.pdf")
    if show:
        plt.show()
    plt.close()

    # ---

    fig, axs = plt.subplots(1, len(BANDS), sharex=True, sharey=True)

    bins = np.linspace(12, 30, 21)

    for i, band in enumerate(BANDS):

        axs[i].hist(
            -2.5 * np.log10(galaxy_table[f"roman_flux_{ROMAN_BAND_KEYS[band]}"]) + ROMAN_BANDPASSES[ROMAN_BAND_KEYS[band]].zeropoint,
            bins=bins,
            histtype="step",
            ec="k",
            ls=":",
            label="True Galaxies",
        )
        axs[i].hist(
            -2.5 * np.log10(pc.list_element(detection_table.filter(flagged)["pgauss_band_flux"], i)) + ROMAN_BANDPASSES[ROMAN_BAND_KEYS[band]].zeropoint,
            bins=bins,
            histtype="step",
            ec="k",
            ls="--",
            label="Flagged Detections",
        )
        axs[i].hist(
            -2.5 * np.log10(pc.list_element(detection_table.filter(~flagged)["pgauss_band_flux"], i)) + ROMAN_BANDPASSES[ROMAN_BAND_KEYS[band]].zeropoint,
            bins=bins,
            histtype="step",
            ec="k",
            label="Unflagged Detections",
        )
        axs[i].set_xlabel(f"{band} [mag]")
        axs[i].set_yscale("log")

    axs[-1].legend(loc='upper left')

    if save:
        plt.savefig(report_path / f"report-mag.png")
        plt.savefig(report_path / f"report-mag.pdf")
    if show:
        plt.show()
    plt.close()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Input directory [str]",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory [str]",
    )
    parser.add_argument(
        "--truth-dir",
        type=str,
        required=True,
        help="Truth directory [str]",
    )
    parser.add_argument(
        "--mosaic",
        type=str,
        required=True,
        help="IMCOM mosaic [str]",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save output",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show output",
    )
    return parser.parse_args()

def main():
    args = get_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    truth_dir = args.truth_dir
    mosaic = args.mosaic
    save = args.save
    show = args.show
    _main(input_dir, output_dir, truth_dir, mosaic, save=save, show=show)

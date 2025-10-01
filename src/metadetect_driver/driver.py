import importlib.metadata
import logging
import math
import sys
from copy import deepcopy
from pathlib import Path

import galsim
import galsim.roman as roman
import metadetect
import ngmix
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import sep
from astropy import wcs
from pyimcom.config import Settings

from .config import parse_driver_config
from .defaults import METADETECT_DEFAULTS

logger = logging.getLogger(__name__)


def _get_package_metadata():
    return {
        f"{__package__} version": importlib.metadata.version(__package__),
        "python version": sys.version,
        "asdf version": importlib.metadata.version("asdf"),
        "astropy version": importlib.metadata.version("astropy"),
        "galsim version": importlib.metadata.version("galsim"),
        "metadetect version": importlib.metadata.version("metadetect"),
        "ngmix version": importlib.metadata.version("ngmix"),
        "numpy version": importlib.metadata.version("numpy"),
        "pyarrow version": importlib.metadata.version("pyarrow"),
        "pyimcom version": importlib.metadata.version("pyimcom"),
        "scipy version": importlib.metadata.version("scipy"),
        "sep version": importlib.metadata.version("sep"),
    }


def write_catalogs(catalogs, base_dir):
    output_path = Path(base_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Writing catalogs to {output_path}")

    # Ensure that all catalogs have the same schema
    schema = None
    for shear_type, catalog in catalogs.items():
        if schema is None:
            schema = catalog.schema
        else:
            _schema = catalog.schema
            assert schema == _schema
    logger.debug(f"Catalog schema is {schema}")

    # TODO update output file naming
    shear_types = catalogs.keys()
    parquet_writers = {}
    for shear_type in shear_types:
        output_file = output_path / f"metadetect_catalog_{shear_type}.parquet"
        logger.debug(f"Opening parquet writer for {shear_type} at {output_file}")
        parquet_writers[shear_type] = pq.ParquetWriter(output_file, schema=_schema)

    for shear_type in shear_types:
        logger.info(f"Writing {shear_type} catalog")
        parquet_writers[shear_type].write(catalogs[shear_type])

    for shear_type in shear_types:
        logger.debug(f"Closing parquet writer for {shear_type}")
        parquet_writers[shear_type].close()

    logger.info("Writing finished")


def run_metadetect(blocks, meta_cfg=None, driver_cfg=None):
    """
    Run metadetect on multi-band coadd images.

    Parameters
    ----------
    blocks : list of OutImage
        PyIMCOM output objects to process. Each element corresponds to the
        coadd of the same block in different bands
    meta_cfg : dict, optional
        Metadetection configuration dictionary. If None, uses default METADETECT_CONFIG. [default : None]
    driver_cfg : dict, optional
        Driver configuration dictionary. If None, uses parsed DEFAULT_EXTRA_CFG. [default : None]

    Returns
    -------
    dict of pyarrow Table
        metadetect catalogs for each shear type
    """
    runner = MetadetectRunner(blocks, meta_cfg=meta_cfg, driver_cfg=driver_cfg)
    return runner.run()


class MetadetectRunner:
    """
    Class to run Metadetection on PyIMCOM coadds (OutImage objects).
    Stores the input coadds, Metadetection config, and driver config, and provides
    methods to build catalogs from the multi-band imaging.
    """

    NATIVE_PIX = Settings.pixscale_native / Settings.arcsec

    def __init__(self, blocks, meta_cfg=None, driver_cfg=None):
        """
        Initialize the MetadetectRunner.

        Parameters
        ----------
        blocks : list of OutImage
            PyIMCOM output objects to process. Each element corresponds to the
            coadd of the same block in different bands
        meta_cfg : dict, optional
            Metadetection configuration dictionary. If None, uses default METADETECT_CONFIG. [default : None]
        driver_cfg : dict, optional
            Driver configuration dictionary. If None, uses parsed DEFAULT_EXTRA_CFG. [default : None]
        """
        logger.info("Instantiating MetadetectRunner")
        logger.debug(f"Metadetect config: {meta_cfg}")
        logger.debug(f"Driver config: {driver_cfg}")

        self.blocks = blocks

        self.meta_cfg = (
            deepcopy(meta_cfg)
            if meta_cfg is not None
            else deepcopy(METADETECT_DEFAULTS)
        )
        self.driver_cfg = parse_driver_config(driver_cfg)
        # parameters (e.g.location center, number of blocks) will be the same.
        # TODO it would be nice to have some way to validate consistency...
        self.cfg = self.blocks[0].cfg
        self.bands = MetadetectRunner.get_bands(self.blocks)
        self.shear_types = self.get_shear_types()
        self.metacal_step = self.get_metacal_step()
        self.det_combs = self.get_det_combs()
        self.shear_combs = self.get_shear_combs()

    def get_metacal_step(self):
        return self.meta_cfg["metacal"].get("step", ngmix.metacal.DEFAULT_STEP)

    def get_shear_types(self):
        return self.meta_cfg["metacal"].get(
            "types", ngmix.metacal.METACAL_MINIMAL_TYPES
        )

    @staticmethod
    def get_bands(blocks):
        """
        Get band names for provided blocks.
        PyIMCOM has a certain ordering of the filters (e.g. filter 2 is H158).
        So, we get the appropiate band name from Settings.RomanFilters (Settings is PyIMCOM settings)

        Returns
        -------
        list[str]
            Band labels matching each block in `blocks`.
        """
        band_list = []
        for block in blocks:
            band = Settings.RomanFilters[block.cfg.use_filter]
            band_list.append(band)
        return band_list

    @staticmethod
    def get_shear(shear_type, metacal_step):
        match shear_type:
            case "noshear":
                _shear = galsim.Shear(g1=0.0, g2=0.0)
            case "1p":
                _shear = galsim.Shear(g1=metacal_step, g2=0)
            case "1m":
                _shear = galsim.Shear(g1=-metacal_step, g2=0)
            case "2p":
                _shear = galsim.Shear(g1=0.0, g2=metacal_step)
            case "2m":
                _shear = galsim.Shear(g1=0.0, g2=-metacal_step)
            case _:
                raise ValueError(f"{shear_type} is an invalid shear type!")

        return _shear

    def get_det_combs(self):
        det_bands = self.driver_cfg["det_bands"]
        if det_bands is not None:
            # Select only detection and shear bands from bands in blocks provided.
            det_idx = np.arange(len(self.bands))[np.isin(self.bands, det_bands)]
            det_combs = [det_idx]
        else:
            det_combs = None

        return det_combs

    def get_shear_combs(self):
        shear_bands = self.driver_cfg["shear_bands"]

        if shear_bands is not None:
            shear_idx = np.arange(len(self.bands))[np.isin(self.bands, shear_bands)]
            shear_combs = [shear_idx]
        else:
            shear_combs = None

        return shear_combs

    @staticmethod
    def get_jacobian(shear_type, metacal_step):
        # cf. https://github.com/GalSim-developers/GalSim/blob/releases/2.7/galsim/gsobject.py#L909-L939
        _shear = MetadetectRunner.get_shear(shear_type, metacal_step)
        return _shear.getMatrix()

    def run(self):
        """
        Main driver to run Metadetection and produce a catalog.

        Parameters
        ----------

        Returns
        -------
        pyarrow Table
            The final combined catalog from all processed blocks for each
            shear type
        """
        mbobs = self.make_mbobs()
        res = self._run_metadetect(mbobs)

        wcs = None
        # Ensure that all blocks have the same WCS
        for block in self.blocks:
            if wcs is None:
                wcs = self.get_wcs(block)
            else:
                _wcs = self.get_wcs(block)
                # TODO is there a better way to check for WCS consistency?
                assert wcs.wcs == _wcs.wcs
        logger.debug(f"WCS is {wcs}")

        return self.construct_table(wcs, res)

    def make_mbobs(self):
        """
        Build an ngmix MultiBandObsList from a list of blocks (each a different band).

        Parameters
        ----------
        blocks : list of OutImage objects

        Returns
        -------
        mbobs : ngmix MultiBandObservation
        """
        mbobs = ngmix.MultiBandObsList()
        for block in self.blocks:
            obslist = self.make_ngmix_obs(block)
            mbobs.append(obslist)
        return mbobs

    def make_ngmix_obs(self, block):
        """
        Create an ngmix ObsList for a single block image.

        Parameters
        ----------
        block : OutImage
            PyIMCOM block

        Returns
        -------
        obslist : ngmix Observation
        """
        img, img_jacobian, psf_img, noise_sigma = self._get_ngmix_data(block)

        # Centers
        psf_cen = (psf_img.shape[0] - 1) / 2.0
        img_cen = (np.array([img.shape[0], img.shape[1]]) - 1) / 2.0

        # ngmix Jacobians
        psf_jac = ngmix.Jacobian(row=psf_cen, col=psf_cen, wcs=img_jacobian)
        img_jac = ngmix.Jacobian(row=img_cen[0], col=img_cen[1], wcs=img_jacobian)

        # Observations
        psf_obs = ngmix.Observation(image=psf_img, jacobian=psf_jac)
        obs = ngmix.Observation(
            image=img,
            jacobian=img_jac,
            weight=np.full(img.shape, 1 / noise_sigma**2, dtype=float),
            psf=psf_obs,
            ormask=np.zeros(img.shape, dtype=np.int32),
            bmask=np.zeros(img.shape, dtype=np.int32),
        )
        obslist = ngmix.ObsList()
        obslist.append(obs)
        return obslist

    def _get_ngmix_data(self, block):
        """
        Generate inputs needed to make ngmix Observation for a single block.

         Parameters
        ----------
        block : OutImage object representing a single block (one band).

        Returns
        -------
        image : np.ndarray
            Coadded image for the requested layer.
        img_jacobian : ngmix.Jacobian
            Image-plane Jacobian derived from WCS at the reference pixel.
        psf_img : np.ndarray
            PSF image.
        noise_sigma : float
            Global RMS of the image background.
        """
        image = block.get_coadded_layer(self.driver_cfg.get("layer", "SCI"))

        # Build GalSim WCS and Jacobian
        w = galsim.AstropyWCS(wcs=self.get_wcs(block))
        img_jacobian = w.jacobian(
            image_pos=galsim.PositionD(w.wcs.wcs.crpix[0], w.wcs.wcs.crpix[1])
        )

        # Estimate background RMS using SEP
        bkg = sep.Background(image.astype(image.dtype.newbyteorder("=")))
        noise_sigma = bkg.globalrms

        # Draw PSF image
        psf_img = self.get_psf(block, w)
        return image, img_jacobian, psf_img, noise_sigma

    def _run_metadetect(self, mbobs):
        """
        Run metadetect on the provided MultiBandObsList.

        Parameters
        ----------
        mbobs : ngmix.MultiBandObsList
            Observations across one or more bands.

        Returns
        -------
        res : dict
            Metadetect results.
        """
        _metadetect_config = deepcopy(self.meta_cfg)
        _rng = np.random.RandomState(seed=self.driver_cfg.get("mdet_seed"))

        return metadetect.do_metadetect(
            _metadetect_config,
            mbobs=mbobs,
            rng=_rng,
            det_band_combs=self.det_combs,
            shear_band_combs=self.shear_combs,
        )

    @staticmethod
    def get_wcs(block):
        """
        Construct WCS for a block from its configuration.

        Parameters
        ----------
        block : OutImage
            PyIMCOM block

        Returns
        -------
        outwcs : astropy.WCS object
            Output WCS from coadded image

        Notes
        -----
        Code mirrors:
        https://github.com/Roman-HLIS-Cosmology-PIT/pyimcom/blob/main/coadd.py
        """
        cfg = block.cfg
        ibx, iby = block.ibx, block.iby
        outwcs = wcs.WCS(naxis=2)
        outwcs.wcs.crpix = [
            (cfg.NsideP + 1) / 2.0 - cfg.Nside * (ibx - (cfg.nblock - 1) / 2.0),
            (cfg.NsideP + 1) / 2.0 - cfg.Nside * (iby - (cfg.nblock - 1) / 2.0),
        ]
        outwcs.wcs.cdelt = [-cfg.dtheta, cfg.dtheta]
        outwcs.wcs.ctype = ["RA---STG", "DEC--STG"]
        outwcs.wcs.crval = [cfg.ra, cfg.dec]
        outwcs.wcs.lonpole = cfg.lonpole
        return outwcs

    @staticmethod
    def get_psf_obj(block):
        """
        Build a GalSim PSF from the block configuration.

        Parameters
        ----------
        block : OutImage
            PyIMCOM block

        Returns
        -------
        psf : galsim object
            Output PSF object

        Notes
        -----
        - PyIMCOM output PSF can be GAUSSIAN, AIRY (obscured/unobscured).
        - The AIRY kernels are also be convolved with a Gaussian.
        """
        cfg = block.cfg

        # Base Gaussian width: cfg.sigmatarget is in native pixels; convert to arcsec then to FWHM.
        fwhm = (
            cfg.sigmatarget
            * MetadetectRunner.NATIVE_PIX
            * 2
            * math.sqrt(2 * math.log(2))
        )
        psf = galsim.Gaussian(fwhm=fwhm)

        # Optional Airy with/without obscuration, then convolve with Gaussian.
        if cfg.outpsf in ("AIRYOBSC", "AIRYUNOBSC"):
            obsc = Settings.obsc if cfg.outpsf == "AIRYOBSC" else 0.0
            # PyIMCOM settings stores the lambda over diameter factor for every band in units of native pixel,
            # so we multiply by roman native pixel scale (0.11) to convert to arcsec
            lam_over_diam = (
                Settings.QFilterNative[cfg.use_filter] * MetadetectRunner.NATIVE_PIX
            )  # arcsec
            airy = galsim.Airy(lam_over_diam=lam_over_diam, obscuration=obsc)
            psf = galsim.Convolve([airy, psf])

        return psf

    def get_psf(self, block, wcs):
        """
        Draw a PSF image for a block given.

        Parameters
        ----------
        block : OutImage
            PyIMCOM block.
        wcs : galsim.BaseWCS
            GalSim WCS instance.

        Returns
        -------
        psf_img: np.ndarray
            PSF image array with shape (psf_img_size, psf_img_size).
        """
        psf = MetadetectRunner.get_psf_obj(block)
        return psf.drawImage(
            nx=self.driver_cfg["psf_img_size"],
            ny=self.driver_cfg["psf_img_size"],
            wcs=wcs,
        ).array

    @staticmethod
    def imcom_flux_conv(flux, dtheta):
        """
        Convert IMCOM flux units to e-/cm^2/s.

        Parameters
        ----------
        flux : np.array
            Array of fluxes

        Returns
        -------
        flux_converted: np.array

        Notes
        -----
        IMCOM flux unit is e- / (0.11 arcsec)^2 / exposure.
        We convert to e-/cm^2/s using Roman collecting area and exposure,
        and correct for the coadd oversampling relative to native pixels.
        The (NATIVE_PIX**2/oversample_pix**2) takes into account that the
        coadds are oversampled and not in the native Roman pixel scale.
        AB magnitude can be calculated using galsim.roman zeropoints.
        """
        # coadd pixel scale in arcsec (PyIMCOM stores in degrees)
        oversample_pix = dtheta * (180.0 / math.pi) * 3600.0  # deg --> arcsec
        norm = (
            roman.exptime
            * roman.collecting_area
            * (MetadetectRunner.NATIVE_PIX**2 / oversample_pix**2)
        )
        return flux / norm

    @staticmethod
    def det_bound_from_padding(cfg):
        """
        Derive a suitable edge bound from the block's padding.
        In this case it sets the bound size to exclude entire padding region.

        Returns
        -------
        int
            Bound size in pixels.
        """
        pad = cfg.postage_pad  # number of postage stamps of padding
        npix_post_stamp = cfg.n2  # pixels per postage stamp
        return int(pad * npix_post_stamp)

    def get_bounded_region(self, res, shear_type):
        """
        Identify detections too close to image edges where detections can be bad.

        Parameters
        ----------
        res : dict
            Metadetect result dict.

        Returns
        -------
        np.ndarray (bool)
            True for detections kept; False for excluded.
        """
        # 'bound_size' sets the maximum distance (in pixels) a detection can be
        # from the edge of the image. If 'bound_size' is None, the boundsize is
        # set to be the padded region from the coadded image.
        if self.driver_cfg["bound_size"] is None:
            bound_size = MetadetectRunner.det_bound_from_padding(self.cfg)
        else:
            bound_size = self.driver_cfg["bound_size"]

        img_size = self.cfg.NsideP  # (ny, nx)
        x = res[shear_type]["sx_col"]
        y = res[shear_type]["sx_row"]
        return (
            (x > bound_size)
            & (x < img_size - bound_size)
            & (y > bound_size)
            & (y < img_size - bound_size)
        )

    def _get_metadata(self):
        _packages = _get_package_metadata()
        # TODO it might be confusing to have `metacal_step` in the metadata
        # if we try to read in the entire catalog through one interface
        # (e.g., via a pyarrow Dataset)
        _meta = {
            "det_band_combs": self.det_combs or "null",
            "shear_band_combs": self.shear_combs or "null",
            "metacal_step": str(self.get_metacal_step()),
        }
        return _packages | _meta

    def construct_table(self, wcs, res):
        """
        Convert metadetect results into pyarrow Tables.
        Converts IMCOM fluxes to e-/cm^2/s,
        computes RA/Dec for detections, and
        checks whether detections are primary (in bounds) or not.

        Parameters
        ----------
        wcs : galsim.WCS
            WCS of coadded image

        res : dict
            Metadetect result dict.

        Returns
        -------
        dict
            dict of pyarrow Tables for each Metadetect catalog

        """
        _metadata = self._get_metadata()
        results = {}
        for shear_type, catalog in res.items():
            _jacobian = MetadetectRunner.get_jacobian(shear_type, self.metacal_step)

            # World coordinates
            w = galsim.AstropyWCS(wcs=wcs)
            x, y = catalog["sx_col"], catalog["sx_row"]
            ra_pos, dec_pos = w.toWorld(x, y, units="deg")

            is_primary = self.get_bounded_region(res, shear_type)

            _results = {}

            _results["is_primary"] = is_primary.tolist()
            _results["ra"] = ra_pos.tolist()
            _results["dec"] = dec_pos.tolist()

            for name in res[shear_type].dtype.names:
                data = catalog[name]
                if ("flux" in name) and ("flags" not in name):
                    data = MetadetectRunner.imcom_flux_conv(data, self.cfg.dtheta)

                _results[name] = data.tolist()

            _results["shear_type"] = [shear_type for _ in range(len(x))]
            _results["jacobian"] = [_jacobian.tolist() for _ in range(len(x))]

            results[shear_type] = pa.Table.from_pydict(_results, metadata=_metadata)

        return results

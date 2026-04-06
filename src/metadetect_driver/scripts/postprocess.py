import argparse
import logging
from pathlib import Path

import healpy as hp
from dustmaps import sfd
from astropy.coordinates import SkyCoord
import healsparse
import galsim.roman
import metadetect_driver
import numpy as np

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import yaml


logging.basicConfig(level=logging.INFO)


flagged = ~(
    pc.field("is_primary")
    & (pc.field("gauss_flags") == 0)
    & (pc.field("pgauss_flags") == 0)
    & (pc.field("psfrec_flags") == 0)
    & (pc.field("gauss_T_flags") == 0)
    & (pc.list_element(pc.field("pgauss_band_flux_flags"), 0) == 0)
    & (pc.list_element(pc.field("pgauss_band_flux_flags"), 1) == 0)
    & (pc.list_element(pc.field("pgauss_band_flux_flags"), 2) == 0)
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

def _process_batch(batch, bands, mask, masked_field, ebv_field, dustmap):

    ra = batch["ra"].to_numpy()
    dec = batch["dec"].to_numpy()

    _masked = mask.get_values_pos(ra, dec, lonlat=True, valid_mask=True)
    masked = pa.array(_masked)
    batch.append_column(masked_field, masked)

    # could be a few fields dependong on which algos run...
    for name in batch.column_names
        if ("flux" in name) and ("flags" not in name):
            field = batch.field(name)
            subarrays = []
            for i in range(num_fields):
                band = bands[i]
                band_key = ROMAN_BAND_KEYS[band]
                bandpass = ROMAN_BANDPASSES[band_key]
                _data = -2.5 * np.log10(pc.list_element(batch[name], i)) + bandpass.zeropoint
                data = pa.array(_data)
                subarrays.append(data)
            mag_name = name.replace("flux", "mag")
            mag_field = pa.field(mag_name, field.type)
            batch.append_column(mag_field, subarrays)

    _coords = SkyCoord(ra, dec, unit="deg")
    _ebv = dustmap.query(_coords)
    ebv = pa.array(_ebv)
    batch.append_column(ebv_field, ebv)

    for name in batch.column_names
        if ("mag" in name) and ("flags" not in name):
            field = batch.field(name)
            subarrays = []
            for i in range(num_fields):
                _data = pc.list_element(batch[name], i) + ebv
                data = pa.array(_data)
                subarrays.append(data)
            dered_name = name + "_dered"
            dered_field = pa.field(dered_name, field.type)
            batch.append_column(ered_field, subarrays)


    batch.drop_columns(
        [
            "is_primary",
            "gauss_flags",
            "pgauss_flags",
            "psfrec_flags",
            "gauss_T_flags",
            "pgauss_band_flux_flags",
        ]
    )

    return batch

def _process_dataset(dataset, bands, mask, dustmap):

    schema = dataset.schema

    # also do flags, reddening, flux to mag, etc.
    masked_field = pa.field('masked', pa.bool_())
    schema = schema.append(masked_field)

    # could be a few fields dependong on which algos run...
    mag_fields = []
    for name in schema.names
        if ("flux" in name) and ("flags" not in name):
            field = schema.field(name)
            mag_name = name.replace("flux", "mag")
            mag_field = pa.field(mag_name, field.type)
            mag_fields.append(mag_field)
            schema = schema.append(mag_field)

    ebv_field = pa.field('ebv', pa.float32())
    schema.append(ebv_field)

    dered_fields = []
    for name in schema.names
        if ("flux" in name) and ("flags" not in name):
            field = schema.field(name)
            dered_name = name + "_dered"
            dered_field = pa.field(dered_name, field.type)
            dered_fields.append(dered_field)
            schema = schema.append(dered_field)

    for field in [
        "is_primary",
        "gauss_flags",
        "pgauss_flags",
        "psfrec_flags",
        "gauss_T_flags",
        "pgauss_band_flux_flags",
    ]:
        index = schema.get_field_index(field)
        schema = schema.remove(index)

    def _batch_iterator(dataset, bands, mask, masked_field, ebv_field, dustmap):
        for batch in dataset.to_batches():
            _processed = _process_batch(batch, bands, mask, masked_field, ebv_field, dustmap)
            yield _processed

    batch_iterator = _batch_iterator(dataset, bands, mask, masked_field, ebv_field)

    return batch_iterator, schema

# https://github.com/des-science/des-y6utils/blob/main/des_y6utils/mdet.py
def postprocess():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--driver-config",
        type=str,
        required=True,
        help="Driver configuration file [yaml]",
    )
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
        "--mask",
        type=str,
        required=False,
        help="HEALSparse mask file [str]",
    )
    args = parser.parse_args()

    driver_config_file = args.driver_config
    input_dir = args.input_dir
    output_dir = args.output_dir
    mask_file = args.mask

    print(f"Loading driver config from {driver_config_file}")
    with open(driver_config_file) as fp:
        driver_config = yaml.safe_load(fp)

    if mask_file is not None:
        mask = healsparse.HealSparseMap.read(mask_file)
    else:
        mask = None

    dataset = ds.dataset(input_dir)

    bands = driver_config["bands"]

    # 1. apply dereddening
    # 2. convert fluxes to magnitudes
    # 3. compute flags & masks
    # 4. partition by shear_type, mosaic, block

    sfd.fetch()
    dustmap = sfd.SFDQuery()

    batch_iterator, schema = _process_dataset(dataset, bands, mask, dustmap)

    # processed_dataset = ds.dataset(
    #     batch_iterator,
    #     schema=schema,
    # )

    ds.write_dataset(
        batch_iterator,
        output_dir,
        format="parquet",
        partitioning=["shear_type", "mosaic"],
        schema=schema,
        existing_data_behavior="delete_matching",
    )

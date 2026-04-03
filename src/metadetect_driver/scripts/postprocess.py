import argparse
import logging
from pathlib import Path

import healpy as hp
import healsparse
import galsim.roman
import metadetect_driver
import numpy as np

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import yaml
# from pyarrow import acero


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

def _process_batch(batch, bands, mask, masked_field):
    masked = mask.get_values_pos(batch["ra"].to_numpy(), batch["dec"].to_numpy(), lonlat=True, valid_mask=True)
    batch.append_column(masked_field, masked)

    return batch

def _process_dataset(dataset, bands, mask):

    schema = dataset.schema

    # also do flags, reddening, flux to mag, etc.
    masked_field = pa.field('masked', pa.bool_())
    schema = schema.append(masked_field)

    def _batch_iterator(dataset, bands, mask, masked_field):
        for batch in dataset.to_batches():
            processed = _process_batch(batch, bands, mask, masked_field)
            yield _processed

    batch_iterator = _batch_iterator(dataset, bands, mask, masked_field)

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

    batch_iterator, schema = _process_dataset(dataset, bands, mask)

    processed_dataset = ds.dataset(
        batch_iterator,
        schema=schema,
    )

    ds.write_dataset(
        processed_dataset
        output_dir,
        partitioning=["shear_type", "mosaic"],
    )

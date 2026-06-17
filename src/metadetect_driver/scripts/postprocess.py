import argparse
import functools
import re
import time

import galsim.roman
import healsparse
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import yaml
from astropy.coordinates import SkyCoord
from dustmaps import sfd

# SourceExtractor names to skip for mag/dered processing
SKIP_NAMES = {"flux", "flux_err", "flux_radius"}

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


def _process_batch(batch, bands, mask, dustmap):
    ra = batch["ra"].to_numpy()
    dec = batch["dec"].to_numpy()

    _masked = mask.get_values_pos(ra, dec, lonlat=True, valid_mask=True)
    masked = pa.array(_masked)
    masked_field = pa.field("masked", pa.bool_())
    batch = batch.append_column(masked_field, masked)

    # could be a few fields dependong on which algos run...
    mag_names = []
    for name in batch.column_names:
        if ("flux" in name) and ("flags" not in name) and (name not in SKIP_NAMES):
            field = batch.field(name)
            _num_fields = field.type.num_fields
            if _num_fields > 0:
                _subarrays = []
                for i in range(_num_fields):
                    band = bands[i]
                    band_key = ROMAN_BAND_KEYS[band]
                    bandpass = ROMAN_BANDPASSES[band_key]
                    _data = -2.5 * np.log10(pc.list_element(batch[name], i)) + bandpass.zeropoint
                    _subarrays.append(_data)
                array = pa.array(zip(*_subarrays))
            else:
                i = int(re.search(r"_(\d+)$", name).group(1))
                band = bands[i]
                band_key = ROMAN_BAND_KEYS[band]
                bandpass = ROMAN_BANDPASSES[band_key]
                _data = -2.5 * np.log10(batch[name]) + bandpass.zeropoint
                array = _data

            mag_name = name.replace("flux", "mag")
            mag_field = pa.field(mag_name, field.type)
            batch = batch.append_column(mag_field, array)
            mag_names.append(mag_name)

    _coords = SkyCoord(ra, dec, unit="deg")
    _ebv = dustmap.query(_coords).astype(np.float64)
    ebv = pa.array(_ebv)
    ebv_field = pa.field("ebv", pa.float64())
    batch = batch.append_column(ebv_field, ebv)

    for name in mag_names:
        field = batch.field(name)
        _num_fields = field.type.num_fields
        if _num_fields > 0:
            _subarrays = []
            for i in range(_num_fields):
                _data = pc.add(pc.list_element(batch[name], i), ebv)
                _subarrays.append(_data)
            array = pa.array(zip(*_subarrays))
        else:
            _data = pc.add(batch[name], ebv)
            array = _data
        dered_name = name + "_dered"
        dered_field = pa.field(dered_name, field.type)
        batch = batch.append_column(dered_field, array)

    is_primary_block = batch["is_primary"]

    # TODO hardcoded for Dec25-sims...
    _a = pc.less(batch["mosaic"], 2)
    _b = pc.less(batch["ra"], 10.3)
    _is_primary_mosaic_ra = pc.or_(
        pc.and_(_a, _b),
        pc.and_(pc.invert(_a), pc.invert(_b)),
    )
    _c = pc.equal(pc.bit_wise_and(batch["mosaic"], 1), 1)
    _d = pc.less(batch["dec"], -43.54188481886891)
    _is_primary_mosaic_dec = pc.or_(
        pc.and_(_c, _d),
        pc.and_(pc.invert(_c), pc.invert(_d)),
    )
    is_primary_mosaic = pc.and_(
        _is_primary_mosaic_ra,
        _is_primary_mosaic_dec,
    )

    is_primary = pc.and_(
        is_primary_block,
        is_primary_mosaic,
    )

    is_primary_block_field = pa.field("is_primary_block", pa.bool_())
    is_primary_mosaic_field = pa.field("is_primary_mosaic", pa.bool_())
    is_primary_field = pa.field("is_primary", pa.bool_())

    batch = batch.drop_columns(["is_primary"])
    batch = batch.append_column(is_primary_block_field, is_primary_block)
    batch = batch.append_column(is_primary_mosaic_field, is_primary_mosaic)
    batch = batch.append_column(is_primary_field, is_primary)

    flag_names = []
    _flagged_columns = []
    for name in batch.column_names:
        if "flags" in name:
            field = batch.field(name)
            _num_fields = field.type.num_fields
            if _num_fields > 0:
                for i in range(_num_fields):
                    _flagged_columns.append(pc.list_element(batch[name], i))
            else:
                _flagged_columns.append(batch[name])
            flag_names.append(name)

    flag_field = pa.field("flagged", pa.bool_())
    _flagged = functools.reduce(pc.bit_wise_or, _flagged_columns)
    flagged = pc.equal(_flagged, 0)
    batch = batch.append_column(flag_field, flagged)

    # TODO: remove uncompressed flags
    # drop_columns = []
    # for name in batch.column_names:
    #     if "flags" in name:
    #         drop_columns.append(name)

    # batch = batch.drop_columns(drop_columns)

    return batch


def _process_dataset(dataset, bands, mask, dustmap):
    schema = dataset.schema

    # also do flags, reddening, flux to mag, etc.
    masked_field = pa.field("masked", pa.bool_())
    schema = schema.append(masked_field)

    # could be a few fields dependong on which algos run...
    mag_names = []
    for name in schema.names:
        if ("flux" in name) and ("flags" not in name) and (name not in SKIP_NAMES):
            field = schema.field(name)
            mag_name = name.replace("flux", "mag")
            mag_field = pa.field(mag_name, field.type)
            schema = schema.append(mag_field)
            mag_names.append(mag_name)

    ebv_field = pa.field("ebv", pa.float64())
    schema = schema.append(ebv_field)

    for name in mag_names:
        field = schema.field(name)
        dered_name = name + "_dered"
        dered_field = pa.field(dered_name, field.type)
        schema = schema.append(dered_field)

    index = schema.get_field_index("is_primary")
    schema = schema.remove(index)
    for name in [
        "is_primary_block",
        "is_primary_mosaic",
        "is_primary",
    ]:
        field = pa.field(name, pa.bool_())
        schema = schema.append(field)

    flag_field = pa.field("flagged", pa.bool_())
    schema = schema.append(flag_field)

    # TODO: remove uncompressed flags
    # for name in schema.names:
    #     if "flags" in name:
    #         index = schema.get_field_index(name)
    #         schema = schema.remove(index)

    def _batch_iterator(dataset, schema, bands, mask, dustmap):
        for batch in dataset.to_batches():
            _processed = _process_batch(batch, bands, mask, dustmap)
            if _processed.schema != schema:
                raise ValueError(
                    f"Realized schema {_processed.schema} not equal to projected schema {schema}!"
                )
            yield _processed

    batch_iterator = _batch_iterator(dataset, schema, bands, mask, dustmap)

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

    start_time = time.time()

    batch_iterator, schema = _process_dataset(dataset, bands, mask, dustmap)

    ds.write_dataset(
        batch_iterator,
        output_dir,
        format="parquet",
        partitioning=["shear_type", "mosaic"],
        partitioning_flavor="hive",
        schema=schema,
        existing_data_behavior="delete_matching",
    )

    end_time = time.time()

    print(f"Finished postprocessing in {end_time - start_time} seconds")

import argparse
import time
from pathlib import Path
import logging

from pyimcom.analysis import OutImage
import pyarrow.parquet as pq

from . import (
    config,
    driver,
)


def get_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("input", type=str, help="input directory")
    # parser.add_argument("output", type=str, help="output directory")
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        help="configuration file [yaml]",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=False,
        default=None,
        help="RNG seed [int]",
    )
    parser.add_argument(
        "--log_level",
        type=int,
        required=False,
        default=2,
        help="logging level [int; 2]",
    )
    return parser.parse_args()


def _write_catalogs(catalogs, output_dir, mosaic, block, coadd_bands):

    coadd_tag = "".join(coadd_bands)

    output_path = Path(output_dir) / f"{coadd_tag}{mosaic}_{block}"
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Writing catalogs to {output_path}")

    # Ensure that all catalogs have the same schema
    schema = None
    for shear_type, catalog in catalogs.items():
        if schema is None:
            schema = catalog.schema
        else:
            _schema = catalog.schema
            assert schema == _schema

    # TODO update output file naming
    shear_types = catalogs.keys()
    parquet_writers = {}
    for shear_type in shear_types:
        output_file = output_path / f"catalog_{shear_type}.parquet"
        print(f"Opening parquet writer to {output_file}")
        parquet_writers[shear_type] = pq.ParquetWriter(output_file, schema=_schema)

    for shear_type in shear_types:
        print(f"Writing {shear_type} catalog")
        parquet_writers[shear_type].write(catalogs[shear_type])

    for shear_type in shear_types:
        print(f"Closing parquet writer to {output_file}")
        parquet_writers[shear_type].close()

    print("Writing finished")


def main():
    args = get_args()

    logging.basicConfig()

    seed = args.seed

    # input_images = args.input

    input_dir = "/hpc/group/cosmology/Roman_HLIS_Cosmology_PIT/Dec25-sims/"
    output_dir = "/work/sdm135/Dec25-sims-mdet/"

    mosaic = "1"  # TODO command line arg
    block = "12_12"  # TODO command line arg and/or iterate?
    coadd_bands = ["Y", "J", "H"]

    input_images = [
        Path(input_dir) / f"{band}{mosaic}_coadds/im3x2-{band}{mosaic}_{block}.cpr.fits.gz"
        for band in coadd_bands
    ]

    print(f"Running metadetect on {input_images}")

    start_time = time.time()

    outimages = [OutImage(input_image) for input_image in input_images]
    results = driver.run_metadetect(
        outimages,
        driver_config=None,
        metadetect_config=None,
        seed=seed,
    )

    end_time = time.time()

    print(f"Finished in {end_time - start_time} seconds")

    print(f"metadetect_stage writing to {output_dir}")

    _write_catalogs(results, output_dir, mosaic, block, coadd_bands)

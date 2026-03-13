import argparse
from pathlib import Path

import pyarrow.parquet as pq

from . import (
    config,
    driver,
)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        type=str,
        help="configuration file [yaml]",
    )
    parser.add_argument("input", type=str, help="output directory")
    parser.add_argument("output", type=str, help="output directory")
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


def _write_catalogs(catalogs, base_dir, block_tag):
    output_path = Path(base_dir)
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
    print(f"Catalog schema is {schema}")

    # TODO update output file naming
    shear_types = catalogs.keys()
    parquet_writers = {}
    for shear_type in shear_types:
        output_file = output_path / f"catalog_{block_tag}_{shear_type}.parquet"
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

    seed = args.seed

    input_images = args.input

    outimages = [OutImage(input_image) for input_image in input_images]
    results = metadetect_driver.run_metadetect(
        outimages,
        config=_driver_config,
        metadetect_config=_metadetect_config,
        seed=seed,
    )

    _output = self.get_output("shear_catalog")
    output = Path(_output)
    print(f"metadetect_stage writing to {output}")

    _write_catalogs(results, output, block_tag)

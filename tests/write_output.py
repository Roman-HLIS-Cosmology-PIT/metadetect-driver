import logging
from pathlib import Path

import metadetect_driver
import pyarrow.parquet as pq
from pyimcom.analysis import OutImage


def _write_catalogs(catalogs, base_dir):
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


def main():
    logging.basicConfig(level=logging.INFO)

    image_paths = [
        str(Path(__file__).parent / "data" / "H158" / "output_00_00.fits"),
        str(Path(__file__).parent / "data" / "J129" / "output_00_00.fits"),
    ]
    print(f"Reading images from {image_paths}")
    outimages = [OutImage(image_path) for image_path in image_paths]

    print("Running metadetect")
    results = metadetect_driver.run_metadetect(outimages)

    output_path = Path(__file__).parent / "output"
    print(f"Writing catalogs to {output_path}")
    _write_catalogs(results, output_path)

    print("Done")


if __name__ == "__main__":
    main()

import argparse
import time
import json
import multiprocessing
import concurrent.futures
from pathlib import Path
import logging

from pyimcom.analysis import OutImage
from pyimcom.compress.compressutils import ReadFile
import pyarrow.parquet as pq
import pyarrow.feather as pf
import yaml

import metadetect_driver


LOG_FORMAT = '%(asctime)s - %(process)d - %(name)s - %(levelname)s - %(message)s'


def task(input_dir, output_dir, coadd_bands, mosaic, block, driver_config=None, metadetect_config=None, seed=None):

    input_images = [
        Path(input_dir) / f"{band}{mosaic}_coadds" / f"im3x2-{band}{mosaic}_{block}.cpr.fits.gz"
        for band in coadd_bands
    ]

    start_time = time.time()

    outimages = [OutImage(input_image) for input_image in input_images]
    results = metadetect_driver.run_metadetect(
        outimages,
        driver_config=driver_config,
        metadetect_config=metadetect_config,
        seed=seed,
    )

    end_time = time.time()

    print(f"Finished block {block} in {end_time - start_time} seconds")

    _write_catalogs(results, output_dir, mosaic, block, coadd_bands)

    return 0


def _write_catalogs(catalogs, output_dir, mosaic, block, coadd_bands):

    coadd_tag = "".join(coadd_bands)

    shear_types = catalogs.keys()

    output_path = Path(output_dir) / f"{coadd_tag}{mosaic}_catalogs"
    output_path.mkdir(parents=True, exist_ok=True)
    # print(f"Writing catalogs to {output_path}")

    for shear_type in shear_types:
        _output_path = output_path / shear_type
        _output_path.mkdir(parents=True, exist_ok=True)

    # # Ensure that all catalogs have the same schema
    # schema = None
    # for shear_type, catalog in catalogs.items():
    #     if schema is None:
    #         schema = catalog.schema
    #     else:
    #         _schema = catalog.schema
    #         assert schema == _schema

    # TODO update output file naming
    for shear_type in shear_types:
        # print(f"Writing {shear_type} catalog")
        # output_file = output_path / shear_type / f"{coadd_tag}{mosaic}_{block}.parquet"
        # pq.write_table(catalogs[shear_type], output_file)
        output_file = output_path / shear_type / f"{coadd_tag}{mosaic}_{block}.feather"
        pf.write_feather(catalogs[shear_type], output_file, compression="uncompressed")  # TODO compression

    # print("Writing finished")


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
        "--mosaic",
        type=str,
        required=True,
        help="IMCOM mosaic [str]",
    )
    parser.add_argument(
        "--driver-config",
        type=str,
        required=False,
        help="Driver configuration file [yaml]",
    )
    parser.add_argument(
        "--metadetect-config",
        type=str,
        required=False,
        help="Metadetect configuration file [yaml]",
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
    parser.add_argument(
        "--njobs",
        type=int,
        required=False,
        default=None,
        help="Number of parallel jobs [int; None]",
    )
    return parser.parse_args()


def get_level(log_level):
    match log_level:
        case 0 | logging.ERROR:
            level = logging.ERROR
        case 1 | logging.WARNING:
            level = logging.WARNING
        case 2 | logging.INFO:
            level = logging.INFO
        case 3 | logging.DEBUG:
            level = logging.DEBUG
        case _:
            level = logging.INFO

    return level


def main():
    args = get_args()

    mp_context = multiprocessing.get_context("forkserver")

    metadetect_config_file = args.metadetect_config
    driver_config_file = args.driver_config
    input_dir = args.input_dir
    output_dir = args.output_dir
    mosaic = args.mosaic
    seed = args.seed
    njobs = args.njobs
    log_level = get_level(args.log_level)

    # Logging doesn't work b/c I haven't setup the handlers for each process
    logging.basicConfig(format=LOG_FORMAT, level=log_level)

    if driver_config_file is not None:
        print(f"Loading driver config from {driver_config_file}")
        with open(driver_config_file) as fp:
            driver_config = yaml.safe_load(fp)
    else:
        driver_config = None

    if metadetect_config_file is not None:
        print(f"Loading metadetect config from {metadetect_config_file}")
        with open(metadetect_config_file) as fp:
            metadetect_config = yaml.safe_load(fp)
    else:
        metadetect_config = None

    _input_file = Path(input_dir) / f"H{mosaic}_coadds"/ f"im3x2-H{mosaic}_00_00.cpr.fits.gz"
    config = ""
    with ReadFile(_input_file) as f:
        for g in f["CONFIG"].data["text"].tolist():
            config += g + " "
        configStruct = json.loads(config)

    nblock = configStruct["BLOCK"]  # NB assume this is always square
    blocks = []
    for i in range(nblock):
        for j in range(nblock):
            _block = f"{i:02d}_{j:02d}"
            blocks.append(_block)

    coadd_bands = ["Y", "J", "H"]

    start_time = time.time()

    with concurrent.futures.ProcessPoolExecutor(max_workers=njobs, mp_context=mp_context) as executor:
        futures = []
        for block in blocks:
            # TODO get new seed...
            _future = executor.submit(
                task,
                input_dir,
                output_dir,
                coadd_bands,
                mosaic,
                block,
                driver_config=driver_config,
                metadetect_config=metadetect_config,
                seed=seed,
            )
            futures.append(_future)

    for future in futures:
        if future.result() != 0:
            print(f"future {id(future)} exited with status {future.result()}")

    end_time = time.time()

    print(f"Finished running all blocks {block} in {end_time - start_time} seconds")



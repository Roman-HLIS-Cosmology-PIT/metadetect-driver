import argparse
import concurrent.futures
import itertools
import json
import logging
import multiprocessing
import os
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml
from pyimcom.analysis import OutImage
from pyimcom.compress.compressutils import ReadFile
from pyimcom.config import Settings

import metadetect_driver

LOG_FORMAT = "%(asctime)s - %(process)d - %(name)s - %(levelname)s - %(message)s"


def _run_metadetect_on_block(
    input_files,
    output_file,
    driver_config,
    metadetect_config,
    seed=None,
):
    start_time = time.time()

    outimages = [OutImage(input_file) for input_file in input_files]
    bands = driver_config["bands"]

    _blocks = [(outimage.ibx, outimage.iby) for outimage in outimages]
    _block_groupby = itertools.groupby(_blocks)
    if not (next(_block_groupby, True) and not next(_block_groupby, False)):
        raise ValueError(f"IMCOM coadds from different blocks ({_blocks})!")

    _mosaics = [outimage.cfg.mosaic for outimage in outimages]
    _mosaic_groupby = itertools.groupby(_mosaics)
    if not (next(_mosaic_groupby, True) and not next(_mosaic_groupby, False)):
        raise ValueError(f"IMCOM coadds from different mosaics ({_mosaics})!")

    _bands = [Settings.RomanFilters[outimage.cfg.use_filter] for outimage in outimages]
    for band, outimage_band in zip(bands, _bands):
        if not outimage_band.startswith(band):
            raise ValueError("IMCOM coadds and driver bands not aligned!")

    results = metadetect_driver.run_metadetect(
        outimages,
        driver_config,
        metadetect_config,
        seed=seed,
    )

    end_time = time.time()

    print(f"Finished block in {end_time - start_time} seconds")

    pq.write_table(results, output_file)

    return 0


def get_log_level(log_level):
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


def run_metadetect_on_images():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-files",
        type=str,
        nargs="+",
        required=True,
        help="Input image files [str]",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Output file [str]",
    )
    # TODO specified in config currently; don't need redundancy
    # parser.add_argument(
    #     "--bands",
    #     type=str,
    #     required=True,
    #     nargs="+",
    #     help="Imaging bands",
    # )
    parser.add_argument(
        "--driver-config",
        type=str,
        required=True,
        help="Driver configuration file [yaml]",
    )
    parser.add_argument(
        "--metadetect-config",
        type=str,
        required=True,
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
    args = parser.parse_args()

    driver_config_file = args.driver_config
    metadetect_config_file = args.metadetect_config
    input_files = args.input_files
    output_file = args.output_file
    seed = args.seed
    log_level = get_log_level(args.log_level)

    # Logging doesn't work b/c I haven't setup the handlers for each process
    logging.basicConfig(format=LOG_FORMAT, level=log_level)

    print(f"Loading driver config from {driver_config_file}")
    with open(driver_config_file) as fp:
        driver_config = yaml.safe_load(fp)

    print(f"Loading metadetect config from {metadetect_config_file}")
    with open(metadetect_config_file) as fp:
        metadetect_config = yaml.safe_load(fp)

    bands = driver_config["bands"]

    if len(input_files) != len(bands):
        raise ValueError(
            f"Number of input images ({len(input_files)}) is not equal to number of bands ({len(bands)})!"
        )
    for input_file in input_files:
        if not os.path.isfile(input_file):
            raise ValueError(f"Input image file ({input_file}) does not exist!")

    start_time = time.time()

    _run_metadetect_on_block(
        input_files,
        output_file,
        driver_config,
        metadetect_config,
        seed=seed,
    )

    end_time = time.time()

    print(f"Finished running block in {end_time - start_time} seconds")


def run_metadetect_on_block():
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
        "--block",
        type=str,
        required=True,
        help="IMCOM block [str; XX_YY]",
    )
    parser.add_argument(
        "--driver-config",
        type=str,
        required=True,
        help="Driver configuration file [yaml]",
    )
    parser.add_argument(
        "--metadetect-config",
        type=str,
        required=True,
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
    args = parser.parse_args()

    driver_config_file = args.driver_config
    metadetect_config_file = args.metadetect_config
    input_dir = args.input_dir
    output_dir = args.output_dir
    mosaic = args.mosaic
    block = args.block
    seed = args.seed
    log_level = get_log_level(args.log_level)

    print(f"Running on mosaic {mosaic}, block {block}")

    # Logging doesn't work b/c I haven't setup the handlers for each process
    logging.basicConfig(format=LOG_FORMAT, level=log_level)

    print(f"Loading driver config from {driver_config_file}")
    with open(driver_config_file) as fp:
        driver_config = yaml.safe_load(fp)

    print(f"Loading metadetect config from {metadetect_config_file}")
    with open(metadetect_config_file) as fp:
        metadetect_config = yaml.safe_load(fp)

    _input_file = Path(input_dir) / f"H{mosaic}_coadds" / f"im3x2-H{mosaic}_{block}.cpr.fits.gz"
    config = ""
    with ReadFile(_input_file) as f:
        for g in f["CONFIG"].data["text"].tolist():
            config += g + " "
        configStruct = json.loads(config)

    bands = driver_config["bands"]

    start_time = time.time()

    input_files = [
        Path(input_dir) / f"{band}{mosaic}_coadds" / f"im3x2-{band}{mosaic}_{block}.cpr.fits.gz"
        for band in bands
    ]

    coadd_tag = "".join(bands)

    output_path = Path(output_dir) / f"{coadd_tag}{mosaic}_catalogs"
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / f"{coadd_tag}{mosaic}_{block}.parquet"

    _run_metadetect_on_block(
        input_files,
        output_file,
        driver_config,
        metadetect_config,
        seed=seed,
    )

    end_time = time.time()

    print(f"Finished running block {block} in {end_time - start_time} seconds")


def run_metadetect_on_mosaic():
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
        required=True,
        help="Driver configuration file [yaml]",
    )
    parser.add_argument(
        "--metadetect-config",
        type=str,
        required=True,
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
    args = parser.parse_args()

    mp_context = multiprocessing.get_context("forkserver")

    driver_config_file = args.driver_config
    metadetect_config_file = args.metadetect_config
    input_dir = args.input_dir
    output_dir = args.output_dir
    mosaic = args.mosaic
    seed = args.seed
    njobs = args.njobs
    log_level = get_log_level(args.log_level)

    rng = np.random.default_rng(seed)
    maxint = 2**32 - 1

    # Logging doesn't work b/c I haven't setup the handlers for each process
    logging.basicConfig(format=LOG_FORMAT, level=log_level)

    print(f"Running on mosaic {mosaic}")

    print(f"Loading driver config from {driver_config_file}")
    with open(driver_config_file) as fp:
        driver_config = yaml.safe_load(fp)

    print(f"Loading metadetect config from {metadetect_config_file}")
    with open(metadetect_config_file) as fp:
        metadetect_config = yaml.safe_load(fp)

    _input_file = Path(input_dir) / f"H{mosaic}_coadds" / f"im3x2-H{mosaic}_00_00.cpr.fits.gz"
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

    bands = driver_config["bands"]
    coadd_tag = "".join(bands)

    output_path = Path(output_dir) / f"{coadd_tag}{mosaic}_catalogs"
    output_path.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    with concurrent.futures.ProcessPoolExecutor(max_workers=njobs, mp_context=mp_context) as executor:
        futures = []
        for block in blocks:
            _seed = rng.integers(0, maxint)
            _input_files = [
                Path(input_dir) / f"{band}{mosaic}_coadds" / f"im3x2-{band}{mosaic}_{block}.cpr.fits.gz"
                for band in bands
            ]
            _output_file = output_path / f"{coadd_tag}{mosaic}_{block}.parquet"
            _future = executor.submit(
                _run_metadetect_on_block,
                _input_files,
                _output_file,
                driver_config,
                metadetect_config,
                seed=_seed,
            )
            futures.append(_future)

    for future in futures:
        if future.result() != 0:
            print(f"future {id(future)} exited with status {future.result()}")

    end_time = time.time()

    print(f"Finished running all blocks in {end_time - start_time} seconds")

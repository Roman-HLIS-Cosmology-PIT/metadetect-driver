import logging
from pathlib import Path

from metadetect_driver import MetaDetectRunner
from pyimcom.analysis import OutImage


def main():
    logging.basicConfig(level=logging.INFO)

    image_paths = [
        str(Path(__file__).parent / "data" / "H158" / "output_00_00.fits"),
        str(Path(__file__).parent / "data" / "J129" / "output_00_00.fits"),
    ]
    print(f"Reading images from {image_paths}")
    outimages = [OutImage(image_path) for image_path in image_paths]

    print(f"Running metadetect")
    mdet_runner = MetaDetectRunner(outimages)
    catalogs = mdet_runner.make_catalogs()

    output_path = Path(__file__).parent / "output"
    print(f"Writing catalogs to {output_path}")
    mdet_runner.write_catalogs(output_path, catalogs)

    print("Done")


if __name__ == "__main__":
    main()

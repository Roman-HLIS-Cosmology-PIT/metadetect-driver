import logging
from pathlib import Path

import metadetect_driver
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
    mdet_runner = metadetect_driver.MetaDetectRunner(outimages)
    catalogs = mdet_runner.run_metadetect()

    output_path = Path(__file__).parent / "output"
    print(f"Writing catalogs to {output_path}")
    metadetect_driver.write_catalogs(catalogs, output_path)

    print("Done")


if __name__ == "__main__":
    main()

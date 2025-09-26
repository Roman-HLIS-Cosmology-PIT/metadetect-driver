from pathlib import Path

from pyimcom.analysis import OutImage
from metadetect_driver import MetaDetectRunner


def main():
    image_path = str(Path(__file__).parent / "data" / "output_00_00.fits")
    print(f"Reading image from {image_path}")
    outimage = OutImage(image_path)

    print(f"Running metadetect")
    mdet_runner = MetaDetectRunner(outimage)
    catalogs, block_indices = mdet_runner.make_catalogs()

    output_path = Path(__file__).parent / "output"
    print(f"Writing catalogs to {output_path}")
    mdet_runner.write_catalogs(output_path, catalogs, block_indices)

    print("Done")

if __name__ == "__main__":
    main()

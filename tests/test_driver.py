from pathlib import Path

import metadetect_driver
import pyarrow.parquet as pq
from pandas.testing import assert_frame_equal
from pyimcom.analysis import OutImage


def test_main():
    image_paths = [
        str(Path(__file__).parent / "data" / "H158" / "output_00_00.fits"),
        str(Path(__file__).parent / "data" / "J129" / "output_00_00.fits"),
    ]
    outimages = [OutImage(image_path) for image_path in image_paths]

    results = metadetect_driver.run_metadetect(outimages)

    output_path = Path(__file__).parent / "output"

    for shear_type, result in results.items():
        expected = pq.read_table(
            output_path / f"metadetect_catalog_{shear_type}.parquet"
        )

        # NOTE this does _not_ require that the metadata be equal, so this will
        # not raise an error even if there is a version mismatch between
        # packages recorded in the metadata
        assert result.schema == expected.schema

        # NOTE we use pandas to check equality because of nan handling.
        # This is not ideal.
        assert_frame_equal(result.to_pandas(), expected.to_pandas())

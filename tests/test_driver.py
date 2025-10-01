from pathlib import Path

import metadetect_driver
import pyarrow as pa
import pyarrow.parquet as pq
from pandas.testing import assert_frame_equal
from pyimcom.analysis import OutImage


def test_main():
    image_paths = [
        str(Path(__file__).parent / "data" / "H158" / "output_00_00.fits"),
        str(Path(__file__).parent / "data" / "J129" / "output_00_00.fits"),
    ]
    outimages = [OutImage(image_path) for image_path in image_paths]

    mdet_runner = metadetect_driver.MetaDetectRunner(outimages)
    catalogs = mdet_runner.run_metadetect()

    output_path = Path(__file__).parent / "output"

    for shear_type, _result in catalogs.items():
        _expected = pq.read_table(
            output_path / f"metadetect_catalog_{shear_type}.parquet"
        )

        # NOTE this does _not_ require that the metadata be equal, so this will
        # not raise an error even if there is a version mismatch between
        # packages recorded in the metadata
        assert _result.schema == _expected.schema

        # NOTE we use pandas to check equality because of nan handling.
        # This is not ideal.
        assert_frame_equal(_result.to_pandas(), _expected.to_pandas())

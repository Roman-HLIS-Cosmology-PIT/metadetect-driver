from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from pandas.testing import assert_frame_equal

from pyimcom.analysis import OutImage
from metadetect_driver import MetaDetectRunner


def test_main():
    image_path = str(Path(__file__).parent / "data" / "output_00_00.fits")
    outimage = OutImage(image_path)

    mdet_runner = MetaDetectRunner(outimage)
    catalogs, block_indices = mdet_runner.make_catalogs()

    output_path = Path(__file__).parent / "output"

    for shear_type in mdet_runner.shear_types:
        _result = pa.concat_tables([cat[shear_type] for cat in catalogs])
        _expected = pq.read_table(output_path / f"metadetect_catalog_{shear_type}.parquet")

        assert _result.schema == _expected.schema, f"Schema for shear_type {shear_type} differ"

        # NOTE we use pandas to check equality because of nan handling.
        # This is not ideal.
        assert_frame_equal(_result.to_pandas(), _expected.to_pandas())

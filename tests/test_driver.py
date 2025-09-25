from pathlib import Path

import pyarrow as pa
# import pyarrow.compute as pc
import pyarrow.parquet as pq
from pandas.testing import assert_frame_equal

from pyimcom.analysis import OutImage
from metadetect_driver import MetaDetectRunner


def test_main():
    image_path = str(Path(__file__).parent / "data" / "output_00_00.fits")
    outimage = OutImage(image_path)

    mdet_runner = MetaDetectRunner(outimage)
    catalogs, block_indices = mdet_runner.make_catalogs()

    # TODO test selections
    # is_primary = pc.field("is_primary") == True
    # is_quality = (
    #     (pc.field("pgauss_s2n") > 10)
    #     & (pc.field("pgauss_T_ratio") > 0.5)
    #     & (pc.field("pgauss_flags") == 0)
    # )

    output_path = Path(__file__).parent / "output"
    # mdet_runner.write_catalogs(output_path, catalogs, block_indices)

    for shear_step in mdet_runner.shear_steps:
        _result = pa.concat_tables([cat[shear_step] for cat in catalogs])
        _expected = pq.read_table(output_path / f"metadetect_catalog_{shear_step}.parquet")
        assert _result.schema == _expected.schema, f"Schema for shear_step {shear_step} differ"
        # NOTE we use pandas to check equality because of nan handling.
        # This is not ideal.
        assert_frame_equal(_result.to_pandas(), _expected.to_pandas())

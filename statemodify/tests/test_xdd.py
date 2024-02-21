import tempfile
import unittest
from pathlib import Path

import pandas as pd
import pkg_resources

import statemodify as stm


class TestXdd(unittest.TestCase):
    def test_convert(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            stm.convert_xdd(
                output_path=tmp_dir,
                xdd_files=pkg_resources.resource_filename(
                    "statemodify", "data/test.xdd"
                ),
                parallel_jobs=1,
            )
            expected = pd.read_parquet(
                pkg_resources.resource_filename("statemodify", "data/test.parquet")
            )
            actual = pd.read_parquet(Path(tmp_dir).joinpath("test.parquet"))

            pd.testing.assert_frame_equal(expected, actual)

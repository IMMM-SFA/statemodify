from pathlib import Path
import pkg_resources
import tempfile
import unittest

import pandas as pd

from statemodify.xdd import XddConverter


class TestXdd(unittest.TestCase):

    def test_convert(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            c = XddConverter(
                output_path=tmp_dir,
                xdd_files=pkg_resources.resource_filename("statemodify", "data/test.xdd"),
                parallel_jobs=1,
            )
            c.convert()
            expected = pd.read_parquet(pkg_resources.resource_filename("statemodify", "data/test.parquet"))
            actual = pd.read_parquet(Path(tmp_dir).joinpath('test.parquet'))

            pd.testing.assert_frame_equal(expected, actual)

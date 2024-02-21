import os
import tempfile
import unittest

import numpy as np
import pandas as pd
import pkg_resources

import statemodify as stm


class TestDdr(unittest.TestCase):
    VALID_MODIFY_DICT = {
        # ids can either be 'struct' or 'id' values
        "ids": ["3600507.01", "3600507.02"],
        "bounds": [0.5, 1.5],
        # turn id on or off completely or for a given period
        # if 0 = off, 1 = on, YYYY = on for years >= YYYY, -YYYY = off for years > YYYY; see file header
        "on_off": [-1977, 1],
        # apply rank of administrative order where 0 is lowest (senior) and n is highest (junior); None is no change
        "admin": [None, 0],
    }

    VALID_MODIFY_DICT_VALUES = {
        # ids can either be 'struct' or 'id' values
        "ids": ["3600507.01", "3600507.02"],
        # turn id on or off completely or for a given period
        # if 0 = off, 1 = on, YYYY = on for years >= YYYY, -YYYY = off for years > YYYY; see file header
        "on_off": [-1977, 1],
        # optionally, pass values that you want to be set for each user group; this overrides bounds
        "values": [0.7],
    }

    DDR_COMP_FILE_NAME = "cm2015B_S0_test.ddr"
    DDR_COMP_FULLPATH = pkg_resources.resource_filename(
        "statemodify", os.path.join("tests/data", DDR_COMP_FILE_NAME)
    )
    SETUP_DF = pd.DataFrame({"id": ["3600507.01", "3600507.02"], "on_off": [0, 1]})
    COMP_DF = pd.DataFrame({"id": ["3600507.01", "3600507.02"], "on_off": [-1977, 1]})
    SETUP_DF_ADMIN = pd.DataFrame(
        {"id": ["3600507.01", "3600507.02"], "admin": ["0.00000", "10.00000"]}
    )
    COMP_DF_ADMIN = pd.DataFrame(
        {"id": ["3600507.01", "3600507.02"], "admin": ["0.00000", "0.00000"]}
    )

    def test_apply_on_off_modification(self):
        """Ensure output matches expected."""

        df = stm.apply_on_off_modification(
            df=TestDdr.SETUP_DF, modify_dict=TestDdr.VALID_MODIFY_DICT, query_field="id"
        )

        pd.testing.assert_frame_equal(TestDdr.COMP_DF, df)

    def test_apply_seniority_modification(self):
        """Ensure output matches expected."""

        df = stm.apply_seniority_modification(
            df=TestDdr.SETUP_DF_ADMIN,
            modify_dict=TestDdr.VALID_MODIFY_DICT,
            query_field="id",
        )

        pd.testing.assert_frame_equal(TestDdr.COMP_DF_ADMIN, df)

    def test_modify_single_ddr_run(self):
        """Ensure the single file processor runs and generates the expected output."""

        with tempfile.TemporaryDirectory() as tmp_dir:
            stm.modify_single_ddr(
                modify_dict=TestDdr.VALID_MODIFY_DICT,
                query_field="id",
                sample=np.array([1.19646919]),
                sample_id=0,
                output_dir=tmp_dir,
                scenario="test",
                basin_name="Upper_Colorado",
                skip_rows=0,
                template_file=None,
            )

            # get contents of comparison file
            with open(TestDdr.DDR_COMP_FULLPATH) as comp:
                comp_data = comp.read()

            # get contents of generated file
            with open(os.path.join(tmp_dir, TestDdr.DDR_COMP_FILE_NAME)) as sim:
                sim_data = sim.read()

            # ensure equality
            self.assertEqual(
                comp_data,
                sim_data,
                msg="Simulated data for DDR file does not match what was expected.",
            )

    def test_modify_ddr_run(self):
        """Ensure the parallel function runs and generates an expected output."""

        with tempfile.TemporaryDirectory() as tmp_dir:
            # generate a batch of files using generated LHS
            stm.modify_ddr(
                modify_dict=TestDdr.VALID_MODIFY_DICT,
                query_field="id",
                output_dir=tmp_dir,
                scenario="test",
                basin_name="Upper_Colorado",
                sampling_method="LHS",
                n_samples=1,
                skip_rows=0,
                n_jobs=1,
                seed_value=123,
                template_file=None,
            )

            # get contents of comparison file
            with open(TestDdr.DDR_COMP_FULLPATH) as comp:
                comp_data = comp.read()

            # get contents of generated file
            with open(os.path.join(tmp_dir, TestDdr.DDR_COMP_FILE_NAME)) as sim:
                sim_data = sim.read()

            # ensure equality
            self.assertEqual(
                comp_data,
                sim_data,
                msg="Simulated data for DDR file does not match what was expected in parallel function.",
            )


if __name__ == "__main__":
    unittest.main()

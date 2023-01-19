import os
import pkg_resources
import tempfile
import unittest

import numpy as np
import pandas as pd

import statemodify as stm


class TestDdr(unittest.TestCase):

    VALID_MODIFY_DICT = {
        # ids can either be 'struct' or 'id' values
        "ids": [["3600507.01", "3600507.02"], ["3600642.04", "3600642.05"]],
        "bounds": [[0.5, 1.5], [0.5, 1.5]],
        # turn id on or off completely or for a given period
        # if 0 = off, 1 = on, YYYY = on for years >= YYYY, -YYYY = off for years > YYYY; see file header
        "on_off": [[-1977, 1], [0, 1977]]
    }

    DDR_COMP_FILE_NAME = "template_scenario-test_sample-0.ddr"
    DDR_COMP_FULLPATH = pkg_resources.resource_filename("statemodify", os.path.join("data", DDR_COMP_FILE_NAME))
    SETUP_DF = pd.DataFrame({"id": ["3600507.01", "3600507.02", "3600642.04", "3600642.07"],
                            "on_off": [0, 1, 1, 1]})
    COMP_DF = pd.DataFrame({"id": ["3600507.01", "3600507.02", "3600642.04", "3600642.07"],
                            "on_off": [-1977, 1, 0, 1]})

    def test_apply_on_off_modification(self):
        """Ensure output matches expected."""

        df = stm.apply_on_off_modification(df=TestDdr.SETUP_DF,
                                           modify_dict=TestDdr.VALID_MODIFY_DICT,
                                           query_field="id")

        pd.testing.assert_frame_equal(TestDdr.COMP_DF, df)




    def test_modify_single_ddr_run(self):
        """Ensure the single file processor runs and generates the expected output."""

        with tempfile.TemporaryDirectory() as tmp_dir:

            stm.modify_single_ddr(modify_dict=TestDdr.VALID_MODIFY_DICT,
                                  query_field="id",
                                  sample=np.array([1.19646919, 0.78613933]),
                                  sample_id=0,
                                  output_dir=tmp_dir,
                                  scenario="test",
                                  skip_rows=0,
                                  template_file=None)

            # get contents of comparison file
            with open(TestDdr.DDR_COMP_FULLPATH) as comp:
                comp_data = comp.read()

            # get contents of generated file
            with open(os.path.join(tmp_dir, TestDdr.DDR_COMP_FILE_NAME)) as sim:
                sim_data = sim.read()

            # ensure equality
            self.assertEqual(comp_data, sim_data, msg="Simulated data for DDM file does not match what was expected.")

    def test_modify_ddr_run(self):
        """Ensure the parallel function runs and generates an expected output."""

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir = "/Users/d3y010/Desktop"

            # generate a batch of files using generated LHS
            stm.modify_ddr(modify_dict=TestDdr.VALID_MODIFY_DICT,
                           query_field="id",
                           output_dir=tmp_dir,
                           scenario="test",
                           sampling_method="LHS",
                           n_samples=1,
                           skip_rows=0,
                           n_jobs=1,
                           seed_value=123,
                           template_file=None)

            # get contents of comparison file
            with open(TestDdr.DDR_COMP_FULLPATH) as comp:
                comp_data = comp.read()

            # get contents of generated file
            with open(os.path.join(tmp_dir, TestDdr.DDR_COMP_FILE_NAME)) as sim:
                sim_data = sim.read()

            # ensure equality
            self.assertEqual(comp_data,
                             sim_data,
                             msg="Simulated data for DDM file does not match what was expected in parallel function.")


if __name__ == '__main__':
    unittest.main()

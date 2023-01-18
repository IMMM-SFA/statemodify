import os
import pkg_resources
import tempfile
import unittest

import numpy as np

import statemodify as stm


class TestDdr(unittest.TestCase):

    VALID_MODIFY_DICT = {
        "ids": [["3600507.01", "3600507.02"], ["3600642.04", "3600642.05"]],
        "struct": ["3600507", "3600603"],
        "bounds": [[0.5, 1.5], [0.5, 1.5]]
    }

    DDR_COMP_FILE_NAME = "template_scenario-test_sample-0.ddr"
    DDR_COMP_FULLPATH = pkg_resources.resource_filename("statemodify", os.path.join("data", DDR_COMP_FILE_NAME))

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

    def test_modify_ddm_run(self):
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

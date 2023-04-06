import os
import pkg_resources
import tempfile
import unittest

import numpy as np

import statemodify as stm


class TestEva(unittest.TestCase):

    VALID_MODIFY_DICT = {
        "ids": [["10001", "10004"], ["10005", "10006"]],
        "bounds": [[-0.5, 1.0], [-0.5, 1.0]]
    }

    EVA_COMP_FILE_NAME = "cm2015_scenario-test_sample-0.eva"
    EVA_COMP_FULLPATH = pkg_resources.resource_filename("statemodify", os.path.join("data", EVA_COMP_FILE_NAME))

    def test_modify_single_eva_run(self):
        """Ensure the single file processor runs and generates the expected output."""

        with tempfile.TemporaryDirectory() as tmp_dir:

            stm.modify_single_eva(modify_dict=TestEva.VALID_MODIFY_DICT,
                                  query_field="id",
                                  sample=np.array([0.54470378, -0.070791]),
                                  sample_id=0,
                                  output_dir=tmp_dir,
                                  scenario="test",
                                  basin_name="Upper_Colorado",
                                  skip_rows=1,
                                  template_file=None)

            # get contents of comparison file
            with open(TestEva.EVA_COMP_FULLPATH) as comp:
                comp_data = comp.read()

            # get contents of generated file
            with open(os.path.join(tmp_dir, TestEva.EVA_COMP_FILE_NAME)) as sim:
                sim_data = sim.read()

            # ensure equality
            self.assertEqual(comp_data, sim_data, msg="Simulated data for EVA file does not match what was expected.")

    def test_modify_eva_run(self):
        """Ensure the parallel function runs and generates an expected output."""

        with tempfile.TemporaryDirectory() as tmp_dir:

            # generate a batch of files using generated LHS
            stm.modify_eva(modify_dict=TestEva.VALID_MODIFY_DICT,
                           query_field="id",
                           output_dir=tmp_dir,
                           scenario="test",
                           basin_name="Upper_Colorado",
                           sampling_method="LHS",
                           n_samples=1,
                           skip_rows=1,
                           n_jobs=1,
                           seed_value=123,
                           template_file=None)

            # get contents of comparison file
            with open(TestEva.EVA_COMP_FULLPATH) as comp:
                comp_data = comp.read()

            # get contents of generated file
            with open(os.path.join(tmp_dir, TestEva.EVA_COMP_FILE_NAME)) as sim:
                sim_data = sim.read()

            # ensure equality
            self.assertEqual(comp_data, sim_data, msg="Simulated data for EVA file does not match what was expected in parallel function.")


if __name__ == '__main__':
    unittest.main()

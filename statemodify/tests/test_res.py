import os
import pkg_resources
import tempfile
import unittest

import statemodify as stm


class TestXbmIwr(unittest.TestCase):

    COMP_FILE_NAME = "gm2015B_scenario-test_sample-0.res"
    COMP_FULLPATH = pkg_resources.resource_filename("statemodify", os.path.join("data", COMP_FILE_NAME))

    def test_modify_xbm_iwr_run(self):
        """Ensure the single file processor runs and generates the expected output."""

        with tempfile.TemporaryDirectory() as tmp_dir:

            # generate a batch of files using generated LHS
            stm.modify_res(output_dir=tmp_dir,
                           scenario="test",
                           basin_name="Gunnison",
                           seed_value=0,
                           n_jobs=1,
                           n_samples=1)

            # get contents of comparison file
            with open(TestXbmIwr.COMP_FULLPATH) as comp:
                comp_data = comp.read()

            # get contents of generated file
            with open(os.path.join(tmp_dir, TestXbmIwr.COMP_FULLPATH)) as sim:
                sim_data = sim.read()

            # ensure equality
            self.assertEqual(comp_data, sim_data, msg="Simulated data for RES file does not match what was expected.")


if __name__ == '__main__':
    unittest.main()

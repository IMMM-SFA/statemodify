import os
import pkg_resources
import tempfile
import unittest

import statemodify as stm


class TestXbmIwr(unittest.TestCase):

    XBM_COMP_FILE_NAME = "cm2015x_S0_test.xbm"
    XBM_COMP_FULLPATH = pkg_resources.resource_filename("statemodify", os.path.join("tests/data", XBM_COMP_FILE_NAME))
    IWR_COMP_FILE_NAME = "cm2015B_S0_test.iwr"
    IWR_COMP_FULLPATH = pkg_resources.resource_filename("statemodify", os.path.join("tests/data", IWR_COMP_FILE_NAME))
    REALIZATION_DIR = pkg_resources.resource_filename("statemodify", os.path.join("tests/data"))

    def test_modify_xbm_iwr_run(self):
        """Ensure the single file processor runs and generates the expected output."""

        with tempfile.TemporaryDirectory() as tmp_dir:

            # generate a batch of files using generated LHS
            stm.modify_xbm_iwr(output_dir=tmp_dir,
                               flow_realizations_directory=TestXbmIwr.REALIZATION_DIR,
                               scenario="test",
                               basin_name="Upper_Colorado",
                               seed_value=0,
                               n_jobs=1,
                               n_samples=1,
                               randomly_select_flow_sample=False,
                               desired_sample_number=0)

            # get contents of comparison file
            with open(TestXbmIwr.XBM_COMP_FULLPATH) as comp:
                xbm_comp_data = comp.read()

            # get contents of generated file
            with open(os.path.join(tmp_dir, TestXbmIwr.XBM_COMP_FULLPATH)) as sim:
                xbm_sim_data = sim.read()

            # ensure equality
            self.assertEqual(xbm_comp_data, xbm_sim_data, msg="Simulated data for XBM file does not match what was expected.")

            # get contents of comparison file
            with open(TestXbmIwr.IWR_COMP_FULLPATH) as comp:
                iwr_comp_data = comp.read()

            # get contents of generated file
            with open(os.path.join(tmp_dir, TestXbmIwr.IWR_COMP_FULLPATH)) as sim:
                iwr_sim_data = sim.read()

            # ensure equality
            self.assertEqual(iwr_comp_data, iwr_sim_data, msg="Simulated data for IWR file does not match what was expected.")


if __name__ == '__main__':
    unittest.main()

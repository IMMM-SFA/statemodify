import os
import pkg_resources
import unittest

import statemodify as stm


class TestExtractXreData(unittest.TestCase):

    def test_extract_xre_data(self):
        xre_file = pkg_resources.resource_filename("statemodify", os.path.join("data/gm2015B.xre"))
        structure_ID = '2803590'  # structure ID for reservoir of interest
        structure_name = 'Blue_Mesa'  # name of the reservoir

        df = stm.extract_xre_data(structure_name=structure_name,
                                  structure_id=structure_ID,
                                  input_file=xre_file)

        self.assertEqual(len(df), 2730)
        self.assertEqual(df.columns.tolist(),
                         ['Res ID', 'ACC', 'Year', 'MO', 'Init. Storage', 'From River By Priority',
                          'From River By Storage',
                          'From River By Other', 'From River By Loss', 'From Carrier By Priority',
                          'From Carrier By Other', 'From Carrier By Loss', 'Total Supply',
                          'From Storage to River For Use',
                          'From Storage to River for Exc', 'From Storage to Carrier for use', 'Total Release', 'Evap',
                          'Seep and Spill', 'EOM Content', 'Target Stor', 'Decree Lim', 'River Inflow', 'River Release',
                          'River Divert',
                          'River by Well', 'River Outflow'])


if __name__ == '__main__':
    unittest.main()

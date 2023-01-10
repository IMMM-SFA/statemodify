import os
import pkg_resources
import unittest

import statemodify as stm


class TestEva(unittest.TestCase):

    def test_something(self):

        # a dictionary to describe what you want to modify and the bounds for the LHS
        setup_dict = {
            "names": ["municipal", "standard"],
            "ids": [["10001", "10004"], ["10005", "10006"]],
            "bounds": [[-1.0, 1.0], [-1.0, 1.0]]
        }

        output_directory = pkg_resources.resource_filename("statemodify", "data")
        scenario = "test"

        # the number of samples you wish to generate
        n_samples = 2

        # seed value for reproducibility if so desired
        seed_value = 123

        # generate a batch of files using generated LHS
        stm.modify_eva(modify_dict=setup_dict,
                       output_dir=output_directory,
                       scenario=scenario,
                       n_samples=n_samples,
                       seed_value=seed_value)


if __name__ == '__main__':
    unittest.main()

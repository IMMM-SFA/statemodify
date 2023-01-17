import pkg_resources
import unittest

import numpy as np

import statemodify.utils as utx
import statemodify.modify as modify


class TestUtils(unittest.TestCase):
    """Tests to ensure functionality and stability of utils.py."""

    TEST_YAML = pkg_resources.resource_filename("statemodify", "data/ddm_data_specification.yml")

    def test_yaml(self):
        """Test YAML return string."""

        d = utx.yaml_to_dict(TestUtils.TEST_YAML)

        file_spec = modify.Modify(comment_indicator=d["comment_indicator"],
                                  data_dict=d["data_dict"],
                                  column_widths=d["column_widths"],
                                  column_alignment=d["column_alignment"],
                                  data_types=d["data_types"],
                                  column_list=d["column_list"],
                                  value_columns=d["value_columns"])

        print(file_spec.value_columns)

    def test_yaml_str(self):
        """Test YAML return string."""

        d = utx.yaml_to_dict(TestUtils.TEST_YAML)

        self.assertEqual(str, type(d["comment_indicator"]), msg="Failure for YAML read to get string type.")
        self.assertEqual("#", d["comment_indicator"], msg="Failure for YAML read to get string value.")

    def test_yaml_dict_of_lists(self):
        """Test YAML return dict of lists."""

        d = utx.yaml_to_dict(TestUtils.TEST_YAML)

        self.assertEqual(dict, type(d["data_dict"]), msg="Failure for YAML read to get dict type.")
        self.assertEqual(list, type(d["data_dict"]["prefix"]), msg="Failure for YAML read to get dict key type list.")

    def test_yaml_dict_of_int(self):
        """Test YAML return dict of int."""

        d = utx.yaml_to_dict(TestUtils.TEST_YAML)

        self.assertEqual(dict, type(d["column_widths"]), msg="Failure for YAML read to get dict type.")
        self.assertEqual(int, type(d["column_widths"]["prefix"]), msg="Failure for YAML read to get dict key type int.")
        self.assertEqual(5, d["column_widths"]["prefix"], msg="Failure for YAML read to get dict key value int.")

    def test_yaml_dict_of_types(self):
        """Test YAML return dict of types."""

        d = utx.yaml_to_dict(TestUtils.TEST_YAML)

        self.assertEqual(dict, type(d["data_types"]), msg="Failure for YAML read to get dict type.")
        self.assertEqual(str, d["data_types"]["prefix"], msg="Failure for YAML read to get dict key type.")
        self.assertEqual(np.float64, d["data_types"]["oct"], msg="Failure for YAML read to get dict key type.")

    def test_yaml_list(self):
        """Test YAML return list."""

        d = utx.yaml_to_dict(TestUtils.TEST_YAML)

        self.assertEqual(list, type(d["column_list"]), msg="Failure for YAML read to get list type.")


if __name__ == '__main__':
    unittest.main()

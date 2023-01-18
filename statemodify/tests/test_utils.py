import pkg_resources
import unittest

import numpy as np

import statemodify.utils as utx


class TestUtils(unittest.TestCase):
    """Tests to ensure functionality and stability of utils.py."""

    TEST_YAML = pkg_resources.resource_filename("statemodify", "data/ddm_data_specification.yml")

    def test_select_template_file(self):
        """Test template file retrieval."""

        template_file_custom = utx.select_template_file("/test/eva_data_specification.yml")

        self.assertEqual("/test/eva_data_specification.yml",
                         template_file_custom,
                         msg="Failure for YAML file selection.")

    def test_select_template_file(self):
        """Test template file retrieval."""

        template_file_custom = utx.select_template_file("/test/template_file.eva")

        self.assertEqual("/test/template_file.eva", template_file_custom, msg="Failure for template file selection.")

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

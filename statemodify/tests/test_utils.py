import unittest

import pandas as pd

import statemodify.utils as utils


class TestUtils(unittest.TestCase):
    """Tests to ensure functionality and stability of utils.py."""

    COMP_RIGHT = "    test"
    COMP_LEFT = "test    "
    COMP_NONE = "test"
    COMP_ZEROS = "0.10"
    COMP_DATA = {'a': [0], 'b': ['alpha  '], 'c': [2.0]}
    COMP_FILE = "/my/output/directory/file_scenario-test_sample-0.txt"
    COMP_STR = "item   focus\n"
    COMP_ADJ_DF = pd.DataFrame({"a": [10., 2.], "b": [-1., -0.2]})
    FIELD_DICT = {"a": [], "b": [], "c": []}
    COLUMN_WIDTHS = {"a": 1, "b": 7, "c": 3}
    COLUMN_LIST = ["a", "b", "c"]
    DATA_TYPES = {"a": int, "b": str, "c": float}

    def test_set_alignment(self):
        """Tests for the set_alignment function."""

        # ensure error is raised when value other than right or left is passed
        with self.assertRaises(AssertionError):
            s = utils.set_alignment(value="test",
                                    n_spaces=0,
                                    align="center")

        # check for left padding (right alignment) of four on value; total length of eight
        s = utils.set_alignment(value="test",
                                n_spaces=4,
                                align="right")

        self.assertEqual(self.COMP_RIGHT, s)
        self.assertEqual(len(self.COMP_RIGHT), len(s))

        # check for right padding (left alignment) of four on value; total length of eight
        s = utils.set_alignment(value="test",
                                n_spaces=4,
                                align="left")

        self.assertEqual(self.COMP_LEFT, s)
        self.assertEqual(len(self.COMP_LEFT), len(s))

        # check for no space of four on value; total length of eight
        s = utils.set_alignment(value="test",
                                n_spaces=0,
                                align="left")

        self.assertEqual(self.COMP_NONE, s)
        self.assertEqual(len(self.COMP_NONE), len(s))

        # check for n_spaces smaller than length of value of four; total length of eight
        s = utils.set_alignment(value="test",
                                n_spaces=-2,
                                align="left")

        self.assertEqual(self.COMP_NONE, s)
        self.assertEqual(len(self.COMP_NONE), len(s))

    def test_pad_with_spaces(self):
        """Tests for the pad_with_spaces function.  Majority of tests accounted for in child function."""

        # ensure error is raised when value length is greater than expected width
        with self.assertRaises(TypeError):
            s = utils.pad_with_spaces(value="  test  ",
                                      expected_width=2,
                                      align="right")

    def test_add_zero_padding(self):
        """Tests for the add_zero_padding function."""

        # ensure zeros are added to the desired precision
        s = utils.add_zero_padding(x="0.1",
                                   precision=2)

        self.assertEqual(self.COMP_ZEROS, s)

    def test_populate_dict(self):
        """Tests for the populate_dict function."""

        # ensure parsing into dict is correct
        d = utils.populate_dict(line="0alpha  2.0\n",
                                field_dict=self.FIELD_DICT,
                                column_widths=self.COLUMN_WIDTHS,
                                column_list=self.COLUMN_LIST,
                                data_types=self.DATA_TYPES)

        self.assertEqual(self.COMP_DATA, d)

    def test_construct_outfile_name(self):
        """Tests for construct_outfile_name function."""

        # ensure function produces what is expected
        f = utils.construct_outfile_name(template_file="/my/template/file/file.txt",
                                         output_directory="/my/output/directory",
                                         scenario="test",
                                         sample_id=0)

        self.assertEqual(self.COMP_FILE, f)

    def test_construct_data_string(self):
        """Tests for construct_data_string function."""

        data = utils.construct_data_string(df=pd.DataFrame({"a": ["item   "],
                                                            "b": ["   focus  "]}),
                                           column_list=["a", "b"],
                                           column_widths={"a": 7, "b": 5},
                                           column_alignment={"a": "left", "b": "right"})

        self.assertEqual(self.COMP_STR, data)

    def test_apply_adjustment_factor(self):
        data = utils.apply_adjustment_factor(data_df=pd.DataFrame({"a": [10, 20],
                                                                   "b": [-1, -2],
                                                                   "c": [0, 1]}),
                                             value_columns=["a", "b"],
                                             query_field="c",
                                             target_ids=[1],
                                             factor=0.1)

        pd.testing.assert_frame_equal(self.COMP_ADJ_DF, data)

    if __name__ == '__main__':
        unittest.main()

import unittest

import pandas as pd

import statemodify.modify as modify


class TestModify(unittest.TestCase):
    """Tests to ensure functionality and stability of modify.py."""

    COMP_RIGHT = "    test"
    COMP_LEFT = "test    "
    COMP_NONE = "test"
    COMP_ZEROS = "0.10"
    COMP_DATA = {'a': [0], 'b': ['alpha  '], 'c': [2.0]}
    COMP_FILE = "/my/output/directory/file_scenario-test_sample-0.txt"
    COMP_STR = "item   focus\n"
    COMP_ADJ_DF_ADD = pd.DataFrame({"a": [10., 20.1], "b": [-1., -1.9]})
    COMP_ADJ_DF_MULTIPLY = pd.DataFrame({"a": [10., 2.], "b": [-1., -0.2]})
    FIELD_DICT = {"a": [], "b": [], "c": []}
    COLUMN_WIDTHS = {"a": 1, "b": 7, "c": 3}
    COLUMN_LIST = ["a", "b", "c"]
    DATA_TYPES = {"a": int, "b": str, "c": float}
    BAD_BOUNDS_LOW = [[-0.5, 1.0], [-0.7, 1.0]]
    BAD_BOUNDS_HIGH = [[-0.5, 1.4], [-0.5, 1.0]]
    BAD_BOUNDS_BOTH = [[-0.5, 1.4], [-0.7, 1.0]]

    def test_set_alignment(self):
        """Tests for the set_alignment function."""

        # ensure error is raised when value other than right or left is passed
        with self.assertRaises(AssertionError):
            s = modify.set_alignment(value="test",
                                     n_spaces=0,
                                     align="center")

        # check for left padding (right alignment) of four on value; total length of eight
        s = modify.set_alignment(value="test",
                                 n_spaces=4,
                                 align="right")

        self.assertEqual(self.COMP_RIGHT, s)
        self.assertEqual(len(self.COMP_RIGHT), len(s))

        # check for right padding (left alignment) of four on value; total length of eight
        s = modify.set_alignment(value="test",
                                 n_spaces=4,
                                 align="left")

        self.assertEqual(self.COMP_LEFT, s)
        self.assertEqual(len(self.COMP_LEFT), len(s))

        # check for no space of four on value; total length of eight
        s = modify.set_alignment(value="test",
                                 n_spaces=0,
                                 align="left")

        self.assertEqual(self.COMP_NONE, s)
        self.assertEqual(len(self.COMP_NONE), len(s))

        # check for n_spaces smaller than length of value of four; total length of eight
        s = modify.set_alignment(value="test",
                                 n_spaces=-2,
                                 align="left")

        self.assertEqual(self.COMP_NONE, s)
        self.assertEqual(len(self.COMP_NONE), len(s))

    def test_pad_with_spaces(self):
        """Tests for the pad_with_spaces function.  Majority of tests accounted for in child function."""

        # ensure error is raised when value length is greater than expected width
        with self.assertRaises(TypeError):
            s = modify.pad_with_spaces(value="  test  ",
                                       expected_width=2,
                                       align="right")

    def test_add_zero_padding(self):
        """Tests for the add_zero_padding function."""

        # ensure zeros are added to the desired precision
        s = modify.add_zero_padding(x="0.1",
                                    precision=2)

        self.assertEqual(self.COMP_ZEROS, s)

    def test_populate_dict(self):
        """Tests for the populate_dict function."""

        # ensure parsing into dict is correct
        d = modify.populate_dict(line="0alpha  2.0\n",
                                 field_dict=self.FIELD_DICT,
                                 column_widths=self.COLUMN_WIDTHS,
                                 column_list=self.COLUMN_LIST,
                                 data_types=self.DATA_TYPES,
                                 replace_dict={})

        self.assertEqual(self.COMP_DATA, d)

    def test_construct_outfile_name(self):
        """Tests for construct_outfile_name function."""

        # ensure function produces what is expected
        f = modify.construct_outfile_name(template_file="/my/template/file/file.txt",
                                          output_directory="/my/output/directory",
                                          scenario="test",
                                          sample_id=0)

        self.assertEqual(self.COMP_FILE, f)

    def test_construct_data_string(self):
        """Tests for construct_data_string function."""

        data = modify.construct_data_string(df=pd.DataFrame({"a": ["item   "],
                                                            "b": ["   focus  "]}),
                                            column_list=["a", "b"],
                                            column_widths={"a": 7, "b": 5},
                                            column_alignment={"a": "left", "b": "right"})

        self.assertEqual(self.COMP_STR, data)

    def test_apply_adjustment_factor_add(self):
        """Ensure add adjustment works."""

        data = modify.apply_adjustment_factor(data_df=pd.DataFrame({"a": [10, 20],
                                                                   "b": [-1, -2],
                                                                   "c": [0, 1]}),
                                              value_columns=["a", "b"],
                                              query_field="c",
                                              target_ids=[1],
                                              factor=0.1,
                                              factor_method="add")

        pd.testing.assert_frame_equal(self.COMP_ADJ_DF_ADD, data)

    def test_apply_adjustment_factor_multiply(self):
        """Ensure multiply adjustment works."""

        data = modify.apply_adjustment_factor(data_df=pd.DataFrame({"a": [10, 20],
                                                                   "b": [-1, -2],
                                                                   "c": [0, 1]}),
                                              value_columns=["a", "b"],
                                              query_field="c",
                                              target_ids=[1],
                                              factor=0.1,
                                              factor_method="multiply")

        pd.testing.assert_frame_equal(self.COMP_ADJ_DF_MULTIPLY, data)

    def test_apply_adjustment_factor_error(self):
        """Ensure error is raised for invalid factor_method."""

        with self.assertRaises(KeyError):
            data = modify.apply_adjustment_factor(data_df=pd.DataFrame({"a": [10, 20],
                                                                       "b": [-1, -2],
                                                                       "c": [0, 1]}),
                                                  value_columns=["a", "b"],
                                                  query_field="c",
                                                  target_ids=[1],
                                                  factor=0.1,
                                                  factor_method="divide")

    def test_validate_bounds_low(self):
        """Raise error for low bound violation."""

        with self.assertRaises(ValueError, msg="Low bounds validation failure."):
            modify.validate_bounds(bounds_list=TestModify.BAD_BOUNDS_LOW)

    def test_validate_bounds_high(self):
        """Raise error for high bound violation."""

        with self.assertRaises(ValueError, msg="High bounds validation failure."):
            modify.validate_bounds(bounds_list=TestModify.BAD_BOUNDS_HIGH)

    def test_validate_bounds_both(self):
        """Raise error for both bound violation."""

        with self.assertRaises(ValueError, msg="High and low bounds validation failure."):
            modify.validate_bounds(bounds_list=TestModify.BAD_BOUNDS_BOTH)

    def test_modify_dataclass(self):
        """Ensure the modify data class is functioning properly."""

        file_spec = modify.Modify(comment_indicator="#",
                                  data_dict={"a": []},
                                  column_widths={"a": 5},
                                  column_alignment={"a": "left"},
                                  data_types={"a": int},
                                  column_list=["a"],
                                  value_columns=["a"])

        self.assertEqual("#", file_spec.comment_indicator)
        self.assertEqual({"a": []}, file_spec.data_dict)
        self.assertEqual(["a"], file_spec.column_list)


if __name__ == '__main__':
    unittest.main()

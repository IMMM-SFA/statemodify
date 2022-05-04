import unittest

import statemodify.utils as utils


class TestUtils(unittest.TestCase):
    """Tests to ensure functionality and stability of utils.py."""

    COMP_RIGHT = "    test"
    COMP_LEFT = "test    "
    COMP_NONE = "test"
    COMP_ZEROS = "0.10"

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
        with self.assertRaises(AssertionError):
            s = utils.pad_with_spaces(value="  test  ",
                                      expected_width=2,
                                      align="right")

    def test_add_zero_padding(self):
        """Tests for the add_zero_padding function."""

        # ensure zeros are added to the desired precision
        s = utils.add_zero_padding(x="0.1",
                                   precision=2)

        self.assertEqual(self.COMP_ZEROS, s)


if __name__ == '__main__':
    unittest.main()

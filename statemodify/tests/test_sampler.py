import unittest

import numpy as np

import statemodify.sampler as sampler


class TestSampler(unittest.TestCase):
    VALID_MODIFY_DICT = {
        "ids": ["10001", "10004"],
        "names": ["variable_0", "variable_1"],
        "bounds": [-1.0, 1.0],
    }

    INVALID_MODIFY_DICT = {"ids": ["10001", "10004"], "bounds": [-1.0, 1.0]}

    VALID_MODIFY_DICT_NONAMES = {"ids": ["10001", "10004"], "bounds": [-1.0, 1.0]}

    PROBLEM_DICT = {
        "num_vars": 2,
        "names": ["variable_0", "variable_1"],
        "bounds": [[-1.0, 1.0], [-1.0, 1.0]],
    }

    def test_validate_modify_dict_fail(self):
        """Ensure validation raises error."""

        with self.assertRaises(KeyError):
            d = sampler.validate_modify_dict(
                TestSampler.INVALID_MODIFY_DICT, fill=False
            )

            del d

    def test_build_problem_dict(self):
        """Ensure expected output is generated."""

        result = sampler.build_problem_dict(TestSampler.VALID_MODIFY_DICT)

        self.assertEqual(
            TestSampler.PROBLEM_DICT, result, msg="Expected output was not achieved."
        )

    def test_generate_samples_outcome(self):
        """Ensure sample generator is functioning correctly."""

        result = sampler.generate_samples(
            problem_dict=TestSampler.PROBLEM_DICT,
            n_samples=2,
            sampling_method="LHS",
            seed_value=123,
        )

        expected_result = np.array(
            [[-0.30353081, 0.55131477], [0.22685145, -0.71386067]]
        )

        np.testing.assert_array_equal(
            np.around(expected_result, 4), np.around(result, 4)
        )

    def test_generate_samples_failure(self):
        """Ensure sample generator is failing correctly."""

        with self.assertRaises(KeyError):
            result = sampler.generate_samples(
                problem_dict=TestSampler.PROBLEM_DICT,
                n_samples=2,
                sampling_method="zzzz",
                seed_value=123,
            )

            del result


if __name__ == "__main__":
    unittest.main()

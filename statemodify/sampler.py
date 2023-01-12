from typing import Union, Dict, List

import numpy as np
from SALib.sample import latin


def validate_modify_dict(modify_dict) -> None:
    """Validate user input modify dictionary to ensure all necessary elements are present.

    :param modify_dict:         Dictionary of parameters to setup the sampler.
    :type modify_dict:          Dict[str, List[Union[str, float]]]

    :returns:                   None
    :rtype:                     None

    """

    required_keys = ("names", "ids", "bounds")

    for key in required_keys:
        if key not in modify_dict:
            raise KeyError(f"Missing the following key in user provided modify dictionary:  '{key}'")


def build_problem_dict(modify_dict) -> dict:
    """Build the problem set from the input modify dictionary provided by the user.

    :param modify_dict:         Dictionary of parameters to setup the sampler.
    :type modify_dict:          Dict[str, List[Union[str, float]]]

    :returns:                   Dictionary ready for use by SALib sample generators
    :rtype:                     dict

    :example:

    .. code-block:: python

        import statemodify as stm

        # a dictionary to describe what you want to modify and the bounds for the LHS
        setup_dict = {
            "names": ["municipal", "standard"],
            "ids": [["10001", "10004"], ["10005", "10006"]],
            "bounds": [[-1.0, 1.0], [-1.0, 1.0]]
        }

        # generate problem dictionary to use with SALib sampling components
        problem_dict = stm.build_problem_dict(modify_dict)

    """

    # ensure all keys are present in the user provided dictionary
    validate_modify_dict(modify_dict)

    return {
        'num_vars': len(modify_dict["names"]),
        'names': modify_dict["names"],
        'bounds': modify_dict["bounds"]
    }


def generate_samples(problem_dict: dict,
                     n_samples: int = 1,
                     sampling_method: str = "LHS",
                     seed_value: Union[None, int] = None) -> np.array:
    """Generate an array of statistically generated samples.

    :param problem_dict:        Dictionary of parameters to setup the sampler.
    :type problem_dict:         Dict[str, List[Union[str, float]]]

    :param sampling_method:     Sampling method.  Uses SALib's implementation (see https://salib.readthedocs.io/en/latest/).
                                Currently supports the following method:  "LHS" for Latin Hypercube Sampling
    :type sampling_method:      str

    :param n_samples:           Number of LHS samples to generate, optional. Defaults to 1.
    :type n_samples:            int, optional

    :param seed_value:          Seed value to use when generating samples for the purpose of reproducibility.
                                Defaults to None.
    :type seed_value:           Union[None, int], optional

    :returns:                   Array of samples
    :rtype:                     np.array

    """

    if sampling_method == "LHS":

        return latin.sample(problem=problem_dict,
                            N=n_samples,
                            seed=seed_value).T

    else:
        raise KeyError(f"Selected sampling method is not currently supported.  Please file a feature request here: https://github.com/IMMM-SFA/statemodify/issues")

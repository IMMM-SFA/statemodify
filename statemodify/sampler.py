import pkg_resources
from typing import Union, Dict, List, Tuple

import numpy as np
from SALib.sample import latin

import statemodify.utils as utx


def validate_modify_dict(modify_dict: Dict[str, List[Union[str, float]]],
                         required_keys: Tuple[str] = ("ids", "bounds", "names"),
                         fill: bool = False) -> dict:
    """Validate user input modify dictionary to ensure all necessary elements are present.

    :param modify_dict:         Dictionary of parameters to setup the sampler.
    :type modify_dict:          Dict[str, List[Union[str, float]]]

    :param required_keys:       Keys required to be present in the input dictionary.
    :type required_keys:        Tuple[str]

    :param fill:                If True, fill in missing names using the index of the ids list. Default False.
    :type fill:                 bool

    :returns:                   Dictionary of validated parameters
    :rtype:                     dict

    """

    # ensure names is in the tuple and add it as the last entry
    required_keys = tuple([i for i in required_keys if i != "names"] + ["names"])

    for key in required_keys:
        if (key not in modify_dict) and (key == "names") and (fill is True):
            print(f"NOTICE: Filling 'names' field for 'modify_dict' with generic names as they are not used in this function.")
            modify_dict["names"] = [f"group_{index}" for index, i in enumerate(modify_dict["ids"])]

        elif key not in modify_dict:
            raise KeyError(f"Missing the following key in user provided modify dictionary:  '{key}'")

    return modify_dict


def build_problem_dict(modify_dict: Dict[str, List[Union[str, float]]],
                       fill: bool = False) -> dict:
    """Build the problem set from the input modify dictionary provided by the user.

    :param modify_dict:         Dictionary of parameters to setup the sampler.
    :type modify_dict:          Dict[str, List[Union[str, float]]]

    :param fill:                If True, fill in missing names using the index of the ids list. Default False.
    :type fill:                 bool

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
        problem_dict = stm.build_problem_dict(modify_dict, fill=False)

    """

    # ensure all keys are present in the user provided dictionary
    modify_dict = validate_modify_dict(modify_dict=modify_dict,
                                       fill=fill)

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
                            seed=seed_value)

    else:
        raise KeyError(f"Selected sampling method is not currently supported.  Please file a feature request here: https://github.com/IMMM-SFA/statemodify/issues")


def generate_sample_all_params(n_samples: int = 1,
                               sampling_method: str = "LHS",
                               seed_value: Union[None, int] = None):
    """Generate samples for all parameters.

    :param sampling_method:     Sampling method.  Uses SALib's implementation (see https://salib.readthedocs.io/en/latest/).
                                Currently supports the following method:  "LHS" for Latin Hypercube Sampling
    :type sampling_method:      str

    :param n_samples:           Number of LHS samples to generate, optional. Defaults to 1.
    :type n_samples:            int, optional

    :param seed_value:          Seed value to use when generating samples for the purpose of reproducibility.
                                Defaults to None.
    :type seed_value:           Union[None, int], optional

    """


    yaml_file = pkg_resources.resource_filename("statemodify", "data/parameter_definitions.yml")
    param_dict = utx.yaml_to_dict(yaml_file)

    problem_dict = {'num_vars': 7,
                    'names': ['mu0',
                              'sigma0',
                              'mu1',
                              'sigma1',
                              'p00',
                              'p11',
                              'evap',
                              'rstorage',
                              'powerplants',
                              'envflows',
                              'oilgas',
                              'tribal',
                              'aspinall',
                              'iwr_multiplier_cm',
                              'iwr_multiplier_gm',
                              'iwr_multiplier_sj',
                              'iwr_multiplier_ym',
                              'iwr_multiplier_wm',
                              'trans_multiplier_cm',
                              'trans_multiplier_gm',
                              'trans_multiplier_sj',
                              'trans_multiplier_ym',
                              'trans_multiplier_wm',
                              'muni_multiplier_cm',
                              'muni_multiplier_gm',
                              'muni_multiplier_sj',
                              'muni_multiplier_ym',
                              'muni_multiplier_wm'],
                    'bounds': [[param_dict["mu0"]["lower"], param_dict["mu0"]["upper"]],
                               [param_dict["sigma0"]["lower"], param_dict["sigma0"]["upper"]],
                               [param_dict["mu1"]["lower"], param_dict["mu1"]["upper"]],
                               [param_dict["sigma1"]["lower"], param_dict["sigma1"]["upper"]],
                               [param_dict["p00"]["lower"], param_dict["p00"]["upper"]],
                               [param_dict["p11"]["lower"], param_dict["p11"]["upper"]],
                               [param_dict["evap"]["lower"], param_dict["evap"]["upper"]],
                               [param_dict["rstorage"]["lower"], param_dict["rstorage"]["upper"]],
                               [param_dict["powerplants"]["lower"], param_dict["powerplants"]["upper"]],
                               [param_dict["envflows"]["lower"], param_dict["envflows"]["upper"]],
                               [param_dict["oilgas"]["lower"], param_dict["oilgas"]["upper"]],
                               [param_dict["tribal"]["lower"], param_dict["tribal"]["upper"]],
                               [param_dict["aspinall"]["lower"], param_dict["aspinall"]["upper"]],
                               [param_dict["iwr_multiplier"]["Upper_Colorado"]["lower"], param_dict["iwr_multiplier"]["Upper_Colorado"]["upper"]],
                               [param_dict["iwr_multiplier"]["Gunnison"]["lower"], param_dict["iwr_multiplier"]["Gunnison"]["upper"]],
                               [param_dict["iwr_multiplier"]["San_Juan"]["lower"], param_dict["iwr_multiplier"]["San_Juan"]["upper"]],
                               [param_dict["iwr_multiplier"]["Yampa"]["lower"], param_dict["iwr_multiplier"]["Yampa"]["upper"]],
                               [param_dict["iwr_multiplier"]["White"]["lower"], param_dict["iwr_multiplier"]["White"]["upper"]],
                               [param_dict["trans_multiplier"]["Upper_Colorado"]["lower"], param_dict["trans_multiplier"]["Upper_Colorado"]["upper"]],
                               [param_dict["trans_multiplier"]["Gunnison"]["lower"], param_dict["trans_multiplier"]["Gunnison"]["upper"]],
                               [param_dict["trans_multiplier"]["San_Juan"]["lower"], param_dict["trans_multiplier"]["San_Juan"]["upper"]],
                               [param_dict["trans_multiplier"]["Yampa"]["lower"], param_dict["trans_multiplier"]["Yampa"]["upper"]],
                               [param_dict["trans_multiplier"]["White"]["lower"], param_dict["trans_multiplier"]["White"]["upper"]],
                               [param_dict["muni_multiplier"]["Upper_Colorado"]["lower"], param_dict["muni_multiplier"]["Upper_Colorado"]["upper"]],
                               [param_dict["muni_multiplier"]["Gunnison"]["lower"], param_dict["muni_multiplier"]["Gunnison"]["upper"]],
                               [param_dict["muni_multiplier"]["San_Juan"]["lower"], param_dict["muni_multiplier"]["San_Juan"]["upper"]],
                               [param_dict["muni_multiplier"]["Yampa"]["lower"], param_dict["muni_multiplier"]["Yampa"]["upper"]],
                               [param_dict["muni_multiplier"]["White"]["lower"], param_dict["muni_multiplier"]["White"]["upper"]]]}

    return generate_samples(problem_dict=problem_dict,
                            n_samples=n_samples,
                            sampling_method=sampling_method,
                            seed_value=seed_value)
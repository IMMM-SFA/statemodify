import inspect

from SALib.sample import latin

from statemodify.eva import modify_eva
from statemodify.ddr import modify_ddr
from statemodify.ddm import modify_ddm


def get_required_arguments(fn) -> list:
    """Get all required arguments for a function as a list."""

    signature = inspect.signature(fn)

    return [k for k, v in signature.parameters.items() if v.default is inspect.Parameter.empty]


def get_arguments_values(fn) -> dict:
    """Get all arguments and their values as a dictionary."""

    signature = inspect.signature(fn)

    return {k: v.default if v.default is not inspect.Parameter.empty else None for k, v in signature.parameters.items()}


def generate_parameters(problem_dict: dict) -> dict:
    """Validate and generate parameters needed for the desired functions."""

    required_problem_set_keys = ("n_samples", "num_vars", "names", "bounds")
    provided_problem_set_keys = problem_dict.keys()
    missing_from_problem_set = set(required_problem_set_keys) - set(provided_problem_set_keys)

    if len(missing_from_problem_set) > 0:
        raise KeyError(f"Missing the following required keys in the problem dictionary: {missing_from_problem_set}")

    # generate LHS sample over the problem set
    param_values = latin.sample(problem_dict, problem_dict["n_samples"])

    # output dictionary of arguments specifications per desired function
    function_parameters = {}

    # generate a dictionary of parameters per function
    parameter_dict = {
        "modify_ddr": get_arguments_values(modify_ddr),
        "modify_eva": get_arguments_values(modify_eva),
        "modify_ddm": get_arguments_values(modify_ddm),
    }

    # required arguements
    required_dict = {
        "modify_ddr": get_required_arguments(modify_ddr),
        "modify_eva": get_required_arguments(modify_eva),
        "modify_ddm": get_required_arguments(modify_ddm),
    }

    # fields available for modification per function
    modification_dict = {
        "modify_ddr": ("ids", "on_off", "admin", "values"),
        "modify_eva": ("ids",),
        "modify_ddm": ("ids",)
    }

    # get function parameters for each component
    function_list = [i for i in problem_dict.keys() if "modify_" in i]

    # ensure required arguments are met
    for i in function_list:

        if i not in parameter_dict:
            raise KeyError(f"No function parameters queued for:  {i}")

        possible_params = parameter_dict[i].keys()
        submitted_params = [x for x in problem_dict[i] if x not in modification_dict[i]]

        # ensure parameter names that were submitted by the user are valid options
        missing_from_possible = set(submitted_params) - set(possible_params)

        if len(missing_from_possible) > 0:
            msg = f"The arguments '{missing_from_possible}' were given by the user for '{i}' but were not in the possible options of: '{possible_params}'"
            raise ValueError(msg)

        # ensure that all required non default value arguements are in what the user provides
        missing_from_required = [x for x in required_dict[i] if x not in submitted_params and x != "modify_dict"]

        if len(missing_from_required) > 0:
            msg = f"The arguments '{missing_from_required}' are required for '{i}' but were not in the those provided by the user: '{submitted_params}'"
            raise ValueError(msg)

        # update arguement dictionary for function with what the user has provided
        parameter_dict[i].update(problem_dict[i])
        function_parameters[i] = parameter_dict[i]

        # get corresponding sample
        fn_index = problem_dict["names"].index(i)
        fn_sample = param_values[:, fn_index]
        function_parameters[i]["sample_array"] = fn_sample

        # build modify dict
        fn_modify_dict = {}
        for x in modification_dict[i]:
            if x in problem_dict[i]:
                fn_modify_dict[x] = problem_dict[i][x]

        # add bounds to modify dict
        fn_modify_dict["bounds"] = problem_dict["bounds"][fn_index]

        # add modify dict to parameters
        function_parameters[i]["modify_dict"] = fn_modify_dict

    return function_parameters


def modify_batch(problem_dict: dict) -> dict:
    """Run multiple modification functions from the same problem set and unified sample.

    :param problem_dict:            Dictionary of options for running multiple functions in batch mode.  Used to
                                    generate the sample and pass other options to participating functions.
                                    See the following example.
    :type problem_dict:             dict

    :return:                        Dictionary of settings and samples for each participating funtion.
    :rtype:                         dict

    :example:

    .. code-block:: python

        import statemodify as stm

        # variables that apply to multiple functions
        output_dir = "<your_desired_directory>"
        basin_name = "Upper_Colorado"
        scenario = "1"
        seed_value = 77

        # problem dictionary
        problem_dict = {
            "n_samples": 2,
            'num_vars': 3,
            'names': ['modify_eva', 'modify_ddr', 'modify_ddm'],
            'bounds': [
                [-0.5, 1.0],
                [0.5, 1.0],
                [0.5, 1.0]
            ],
            # additional settings for each function
            "modify_eva": {
                "seed_value": seed_value,
                "output_dir": output_dir,
                "scenario": scenario,
                "basin_name": basin_name,
                "query_field": "id",
                "ids": ["10001", "10004"]
            },
            "modify_ddr": {
                "seed_value": seed_value,
                "output_dir": output_dir,
                "scenario": scenario,
                "basin_name": basin_name,
                "query_field": "id",
                "ids": ["3600507.01", "3600507.02"],
                "admin": [None, 0],
                "on_off": [-1977, 1]
            },
            "modify_ddm": {
                "seed_value": seed_value,
                "output_dir": output_dir,
                "scenario": scenario,
                "basin_name": basin_name,
                "query_field": "id",
                "ids": ["3600507", "3600603"]
            }
        }

        # run in batch
        fn_parameter_dict = stm.modify_batch(problem_dict=problem_dict)

    """

    # functions availabe for use in batch mode
    available_functions = ("modify_eva", "modify_ddm", "modify_ddr")

    # generate validated dictionary of parameters for each modification function
    parameter_dict = generate_parameters(problem_dict=problem_dict)

    for i in parameter_dict.keys():

        print(f"Running {i}")
        params = parameter_dict[i]

        if i == "modify_eva":

            modify_eva(
                modify_dict=params["modify_dict"],
                query_field=params["query_field"],
                output_dir=params["output_dir"],
                scenario=params["scenario"],
                basin_name=params["basin_name"],
                sampling_method=params["sampling_method"],
                n_samples=params["n_samples"],
                skip_rows=params["skip_rows"],
                n_jobs=params["n_jobs"],
                seed_value=params["seed_value"],
                template_file=params["template_file"],
                factor_method=params["factor_method"],
                data_specification_file=params["data_specification_file"],
                min_bound_value=params["min_bound_value"],
                max_bound_value=params["max_bound_value"],
                save_sample=params["save_sample"],
                sample_array=params["sample_array"]
            )

        elif i == "modify_ddr":

            modify_ddr(
                modify_dict=params["modify_dict"],
                query_field=params["query_field"],
                output_dir=params["output_dir"],
                scenario=params["scenario"],
                basin_name=params["basin_name"],
                sampling_method=params["sampling_method"],
                n_samples=params["n_samples"],
                skip_rows=params["skip_rows"],
                n_jobs=params["n_jobs"],
                seed_value=params["seed_value"],
                template_file=params["template_file"],
                factor_method=params["factor_method"],
                data_specification_file=params["data_specification_file"],
                min_bound_value=params["min_bound_value"],
                max_bound_value=params["max_bound_value"],
                save_sample=params["save_sample"],
                sample_array=params["sample_array"]
            )

        elif i == "modify_ddm":

            modify_ddm(
                modify_dict=params["modify_dict"],
                query_field=params["query_field"],
                output_dir=params["output_dir"],
                scenario=params["scenario"],
                basin_name=params["basin_name"],
                sampling_method=params["sampling_method"],
                n_samples=params["n_samples"],
                skip_rows=params["skip_rows"],
                n_jobs=params["n_jobs"],
                seed_value=params["seed_value"],
                template_file=params["template_file"],
                factor_method=params["factor_method"],
                data_specification_file=params["data_specification_file"],
                min_bound_value=params["min_bound_value"],
                max_bound_value=params["max_bound_value"],
                save_sample=params["save_sample"],
                sample_array=params["sample_array"]
            )

        else:
            raise KeyError(
                f"Function '{i}' is not supported in batch mode.  Available functions are:  {available_functions}")

    return parameter_dict

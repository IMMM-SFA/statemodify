import os
from typing import Union, Dict, List

import numpy as np
from joblib import Parallel, delayed

import statemodify.modify as modify
import statemodify.sampler as sampler
import statemodify.utils as utx


def modify_single_eva(modify_dict: Dict[str, List[Union[str, float]]],
                      query_field: str,
                      output_dir: str,
                      scenario: str,
                      basin_name: str,
                      sample: np.array,
                      sample_id: int = 0,
                      skip_rows: int = 1,
                      template_file: Union[None, str] = None,
                      factor_method: str = "add",
                      data_specification_file: Union[None, str] = None,
                      min_bound_value: float = -0.5,
                      max_bound_value: float = 1.0) -> None:
    """Modify StateMod net reservoir evaporation annual data file (.eva) using a Latin Hypercube Sample from the user.
    Samples are processed in parallel. Modification is targeted at ids chosen by the user to
    modify and specified in the `modify_dict` argument.  The user must specify bounds for each field name.

    :param modify_dict:                 Dictionary of parameters to setup the sampler.  See following example.
    :type modify_dict:                  Dict[str, List[Union[str, float]]]

    :param query_field:                 Field name to use for target query.
    :type query_field:                  str

    :param sample:                      An array of samples for each parameter.
    :type sample:                       np.array

    :param sample_id:                   Numeric ID of sample that is being processed. Defaults to 0.
    :type sample_id:                    int

    :param output_dir:                  Path to output directory.
    :type output_dir:                   str

    :param scenario:                    Scenario name.
    :type scenario:                     str

    :param basin_name:                      Name of basin for either:
                                                Upper_Colorado
                                                Yampa
                                                San_Juan
                                                Gunnison
                                                White
    :type basin_name:                       str

    :param skip_rows:                   Number of rows to skip after the commented fields end; default 1
    :type skip_rows:                    int, optional

    :param template_file:               If a full path to a template file is provided it will be used.  Otherwise, the
                                        default template in this package will be used.
    :type template_file:                Union[None, str]

    :param factor_method:               Method by which to apply the factor. Options 'add', 'multiply'.
                                        Defaults to 'add'.
    :type factor_method:                str

    :param data_specification_file:     If a full path to a data specification template is provided it will be used.
                                        Otherwise, the default file in the package is used.
    :type data_specification_file:      Union[None, str]

    :param min_bound_value:             Minimum feasible sampling bounds in feet per month.
                                        Minimum allowable value:  -0.5
    :type min_bound_value:              float

    :param max_bound_value:             Maximum feasible sampling bounds in feet per month.
                                        Maximum allowable value:  1.0
    :type max_bound_value:              float

    :return: None
    :rtype: None

    :example:

    .. code-block:: python

        import statemodify as stm

        # a dictionary to describe what you want to modify and the bounds for the LHS
        setup_dict = {
            "ids": ["10001", "10004"],
            "bounds": [-0.5, 1.0]
        }

        output_directory = "<your desired output directory>"
        scenario = "<your scenario name>"

        # sample id for the current run
        sample_id = 0

        # sample array for each parameter
        sample = np.array([0.39])

        # seed value for reproducibility if so desired
        seed_value = None

        # number of rows to skip in file after comment
        skip_rows = 1

        # name of field to query
        query_field = "id"

        # number of jobs to launch in parallel; -1 is all but 1 processor used
        n_jobs = -1

        # basin to process
        basin_name = "Upper_Colorado"

        # generate a batch of files using generated LHS
        stm.modify_single_eva(modify_dict=modify_dict,
                              query_field=query_field,
                              sample=sample,
                              sample_id=sample_id,
                              output_dir=output_dir,
                              scenario=scenario,
                              basin_name=basin_name,
                              skip_rows=skip_rows,
                              template_file=None,
                              factor_method="add",
                              data_specification_file=None,
                              min_bound_value=-0.5,
                              max_bound_value=1.0)

    """

    # select the appropriate template file
    template_file = utx.select_template_file(basin_name, template_file, extension="eva")

    # read in data specification yaml
    data_specification_file = utx.select_data_specification_file(yaml_file=data_specification_file,
                                                                 extension="eva")
    data_spec_dict = utx.yaml_to_dict(data_specification_file)

    # instantiate data specification and validation class
    file_spec = modify.Modify(comment_indicator=data_spec_dict["comment_indicator"],
                              data_dict=data_spec_dict["data_dict"],
                              column_widths=data_spec_dict["column_widths"],
                              column_alignment=data_spec_dict["column_alignment"],
                              data_types=data_spec_dict["data_types"],
                              column_list=data_spec_dict["column_list"],
                              value_columns=data_spec_dict["value_columns"])

    # prepare template data frame for alteration
    template_df, template_header = modify.prep_data(field_dict=file_spec.data_dict,
                                                    template_file=template_file,
                                                    column_list=file_spec.column_list,
                                                    column_widths=file_spec.column_widths,
                                                    data_types=file_spec.data_types,
                                                    comment=file_spec.comment_indicator,
                                                    skip_rows=skip_rows)

    # strip the query field of any whitespace
    template_df[query_field] = template_df[query_field].str.strip()

    # validate user provided sample bounds to ensure they are within a feasible range
    modify.validate_bounds(bounds_list=[modify_dict["bounds"]],
                           min_value=min_bound_value,
                           max_value=max_bound_value)

    # extract target ids to modify
    id_list = modify_dict["ids"]

    # apply adjustment
    template_df[file_spec.value_columns] = modify.apply_adjustment_factor(data_df=template_df,
                                                                          value_columns=file_spec.value_columns,
                                                                          query_field=query_field,
                                                                          target_ids=id_list,
                                                                          factor=sample,
                                                                          factor_method=factor_method)

    # reconstruct precision
    template_df[file_spec.value_columns] = template_df[file_spec.value_columns].round(4)

    # convert all fields to str type
    template_df = template_df.astype(str)

    # add formatted data to output string
    data = modify.construct_data_string(template_df,
                                        file_spec.column_list,
                                        file_spec.column_widths,
                                        file_spec.column_alignment)

    # write output file
    output_file = modify.construct_outfile_name(template_file, output_dir, scenario, sample_id)

    with open(output_file, "w") as out:
        # write header
        out.write(template_header)

        # write data
        out.write(data)


def modify_eva(modify_dict: Dict[str, List[Union[str, float]]],
               query_field: str,
               output_dir: str,
               scenario: str,
               basin_name: str,
               sampling_method: str = "LHS",
               n_samples: int = 1,
               skip_rows: int = 1,
               n_jobs: int = -1,
               seed_value: Union[None, int] = None,
               template_file: Union[None, str] = None,
               factor_method: str = "add",
               data_specification_file: Union[None, str] = None,
               min_bound_value: float = -0.5,
               max_bound_value: float = 1.0,
               save_sample: bool = False,
               sample_array: Union[None, np.array] = None) -> None:
    """Modify StateMod net reservoir evaporation annual data file (.eva) using a Latin Hypercube Sample from the user.
    Samples are processed in parallel. Modification is targeted at ids chosen by the user to
    modify and specified in the `modify_dict` argument.  The user must specify bounds for sampling.

    :param modify_dict:         Dictionary of parameters to setup the sampler.  See following example.
    :type modify_dict:          Dict[str, List[Union[str, float]]]

    :param query_field:         Field name to use for target query.
    :type query_field:          str

    :param output_dir:          Path to output directory.
    :type output_dir:           str

    :param scenario:            Scenario name.
    :type scenario:             str

    :param basin_name:                      Name of basin for either:
                                                Upper_Colorado
                                                Yampa
                                                San_Juan
                                                Gunnison
                                                White
    :type basin_name:                       str

    :param sampling_method:     Sampling method.  Uses SALib's implementation (see https://salib.readthedocs.io/en/latest/).
                                Currently supports the following method:  "LHS" for Latin Hypercube Sampling
    :type sampling_method:      str

    :param n_samples:           Number of LHS samples to generate, optional. Defaults to 1.
    :type n_samples:            int, optional

    :param skip_rows:           Number of rows to skip after the commented fields end; default 1
    :type skip_rows:            int, optional

    :param n_jobs:              Number of jobs to process in parallel.  Defaults to -1 meaning all but 1 processor.
    :type n_jobs:               int

    :param seed_value:          Seed value to use when generating samples for the purpose of reproducibility.
                                Defaults to None.
    :type seed_value:           Union[None, int], optional

    :param template_file:       If a full path to a template file is provided it will be used.  Otherwise the
                                default template in this package will be used.
    :type template_file:        Union[None, str]

    :param factor_method:       Method by which to apply the factor. Options 'add', 'multiply'.
                                Defaults to 'add'.
    :type factor_method:        str

    :param data_specification_file:     If a full path to a data specification template is provided it will be used.
                                        Otherwise, the default file in the package is used.
    :type data_specification_file:      Union[None, str]

    :param min_bound_value:             Minimum feasible sampling bounds in feet per month.
                                        Minimum allowable value:  -0.5
    :type min_bound_value:              float

    :param max_bound_value:             Maximum feasible sampling bounds in feet per month.
                                        Maximum allowable value:  1.0
    :type max_bound_value:              float

    :param save_sample:         Choice to save LHS sample or not; default False.  If True, sample array will be written
                                to the output directory.
    :type save_sample:          bool

    :param sample_array:        Optionally provide array containing sample instead of generating it.
    :type sample_array:         np.array

    :return: None
    :rtype: None

    :example:

    .. code-block:: python

        import statemodify as stm

        # a dictionary to describe what you want to modify and the bounds for the LHS
        setup_dict = {
            "ids": ["10001", "10004"],
            "bounds": [-0.5, 1.0]
        }

        output_directory = "<your desired output directory>"
        scenario = "<your scenario name>"

        # the number of samples you wish to generate
        n_samples = 4

        # seed value for reproducibility if so desired
        seed_value = None

        # number of rows to skip in file after comment
        skip_rows = 1

        # name of field to query
        query_field = "id"

        # number of jobs to launch in parallel; -1 is all but 1 processor used
        n_jobs = -1

        # basin to process
        basin_name = "Upper_Colorado"

        # generate a batch of files using generated LHS
        stm.modify_eva(modify_dict=setup_dict,
                       query_field=query_field,
                       output_dir=output_directory,
                       scenario=scenario,
                       basin_name=basin_name,
                       sampling_method="LHS",
                       n_samples=n_samples,
                       skip_rows=skip_rows,
                       n_jobs=n_jobs,
                       seed_value=seed_value,
                       template_file=None,
                       factor_method="add",
                       data_specification_file=None,
                       min_bound_value=-0.5,
                       max_bound_value=1.0,
                       save_sample=False)

    """

    if sample_array is None:

        # build a problem dictionary for use by SALib
        problem_dict = sampler.build_problem_dict(modify_dict, fill=True)

        # generate a sample array
        sample_array = sampler.generate_samples(problem_dict=problem_dict,
                                                n_samples=n_samples,
                                                sampling_method=sampling_method,
                                                seed_value=seed_value)

    # if the user chooses, write the sample to file
    if save_sample:
        sample_file = os.path.join(output_dir, f"eva_{n_samples}-samples_scenario-{scenario}.npy")
        np.save(sample_file, sample_array)

    # generate all files in parallel
    results = Parallel(n_jobs=n_jobs, backend="loky")(delayed(modify_single_eva)(modify_dict=modify_dict,
                                                                                 query_field=query_field,
                                                                                 sample=sample,
                                                                                 sample_id=sample_id,
                                                                                 output_dir=output_dir,
                                                                                 scenario=scenario,
                                                                                 basin_name=basin_name,
                                                                                 skip_rows=skip_rows,
                                                                                 template_file=template_file,
                                                                                 factor_method=factor_method,
                                                                                 data_specification_file=data_specification_file,
                                                                                 min_bound_value=min_bound_value,
                                                                                 max_bound_value=max_bound_value) for sample_id, sample in enumerate(sample_array))

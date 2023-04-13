import os
import pkg_resources
from typing import Union, List

import numpy as np
from joblib import Parallel, delayed

import statemodify.sampler as sampler
import statemodify.utils as utx


def get_reservoir_structure_ids(basin_name: str,
                                template_file: Union[None, str] = None,
                                data_specification_file: Union[None, str] = None):
    """Generate a list of structure ids that are in the input file.

    :param basin_name:                      Name of basin for either:
                                                Upper_Colorado
                                                Yampa
                                                San_Juan
                                                Gunnison
                                                White
    :type basin_name:                       str

    :param template_file:       If a full path to a template file is provided it will be used.  Otherwise the
                                default template in this package will be used.
    :type template_file:        Union[None, str]

    :param data_specification_file:     If a full path to a data specification template is provided it will be used.
                                        Otherwise, the default file in the package is used.
    :type data_specification_file:      Union[None, str]

    :returns:                           List of structure ids
    :rtype:                             List

    """

    # select the appropriate template file
    template_file = utx.select_template_file(basin_name, template_file, extension="res")

    # read in data specification yaml
    data_specification_file = utx.select_data_specification_file(yaml_file=data_specification_file,
                                                                 extension="res")
    data_spec_dict = utx.yaml_to_dict(data_specification_file)

    content = []
    with open(template_file) as get:
        for line in get:
            if line[0] == data_spec_dict["comment_indicator"]:
                pass
            else:
                x = line[0:data_spec_dict["id_column_width"]].strip()
                if len(x) > 0:
                    content.append(x)

    return content


def modify_single_res(output_dir: str,
                      scenario: str,
                      basin_name: str,
                      sample: np.array,
                      sample_id: int = 0,
                      template_file: Union[None, str] = None,
                      data_specification_file: Union[None, str] = None,
                      target_structure_id_list: Union[None, List[str]] = None,
                      skip_rows: int = 0):
    """Modify a single reservoir (.res) file based on a user provided sample.

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

    :param sample:              An array of samples for each parameter.
    :type sample:               np.array

    :param sample_id:           Numeric ID of sample that is being processed. Defaults to 0.
    :type sample_id:            int

    :param template_file:       If a full path to a template file is provided it will be used.  Otherwise the
                                default template in this package will be used.
    :type template_file:        Union[None, str]

    :param data_specification_file:     If a full path to a data specification template is provided it will be used.
                                        Otherwise, the default file in the package is used.
    :type data_specification_file:      Union[None, str]

    :param target_structure_id_list:    Structure id list to process.  If None, all structure ids will be processed.
    :type target_structure_id_list:     Union[None, List[str]]

    :param skip_rows:           Number of rows to skip after the commented fields end; default 1
    :type skip_rows:            int, optional

    """

    # select the appropriate template file
    template_file = utx.select_template_file(basin_name, template_file, extension="res")

    # read in data specification yaml
    data_specification_file = utx.select_data_specification_file(yaml_file=data_specification_file,
                                                                 extension="res")
    data_spec_dict = utx.yaml_to_dict(data_specification_file)

    column_widths = data_spec_dict["column_start_index"]

    if target_structure_id_list is None:
        target_structure_id_list = get_reservoir_structure_ids(basin_name=basin_name,
                                                               template_file=template_file,
                                                               data_specification_file=data_specification_file)

    # do not modify any in no modify list
    ignore_parts = data_spec_dict["ignore_structure_with_content"]
    target_structure_id_list = [i for i in target_structure_id_list if f"_{i.split('_')[-1]}" not in ignore_parts]

    capture = False
    account_capture = False
    content = []
    with open(template_file) as get:
        for index, line in enumerate(get):

            # capture header
            if line[0] == data_spec_dict["comment_indicator"] or skip_rows > 0:
                content.append(line)

            # account for any additional non-commented lines that need to be skipped before starting data collection
            elif skip_rows > 0:
                content.append(line)
                skip_rows -= 1

            else:

                # get segment of line that usually houses structure id
                structure_id_segement = line[0:data_spec_dict["id_column_width"]].strip()

                # if reservoir start line found trigger capture
                if (len(structure_id_segement) > 0) and (capture is False) and (data_spec_dict["terminate_string"] not in line):

                    # keep content
                    content.append(line)

                    # set to capture next lines
                    capture = True

                    # assign structure id for current reservoir
                    structure_id = structure_id_segement

                # if capturing line
                elif (len(structure_id_segement) == 0) and (capture is True) and (data_spec_dict["terminate_string"] not in line) and (
                        structure_id in target_structure_id_list):

                    # line contains the
                    if account_capture is False:

                        # get original value
                        volmax_initial = line[column_widths["volmin"]:column_widths["volmax"]]
                        volmax_initial_length = len(volmax_initial)

                        # modify the original value with the multiplier from the sample
                        volmax_modified = f"{int(float(volmax_initial) * sample)}."

                        # reapply space padding
                        n_space_padding = volmax_initial_length - len(volmax_modified)
                        volmax_modified = f"{' ' * n_space_padding}{volmax_modified}"

                        # place back in string
                        line = line[:column_widths["volmin"]] + volmax_modified + line[column_widths["volmax"]:]
                        content.append(line)

                        # initialize list for all accounts for follow
                        account_list = []

                        account_capture = True

                    else:
                        account_list.append(line)

                elif (len(structure_id_segement) == 0) and (capture is True) and (data_spec_dict["terminate_string"] in line) and (
                        structure_id in target_structure_id_list):

                    # compile accounts
                    for acct_line in account_list:
                        # get original values for ownmax and sto-1
                        ownmax_initial = acct_line[column_widths["ownname"]:column_widths["ownmax"]]
                        ownmax_initial_length = len(ownmax_initial)

                        sto_initial = acct_line[column_widths["ownmax"]:column_widths["sto_1"]]
                        sto_initial_length = len(sto_initial)

                        # break out account volume by total number of accounts in equal shares
                        ownmax_modified = f"{int(float(ownmax_initial) * sample)}."
                        sto_modified = f"{int(float(sto_initial) * sample)}."

                        # reapply space padding
                        n_space_padding = ownmax_initial_length - len(ownmax_modified)
                        ownmax_modified = f"{' ' * n_space_padding}{ownmax_modified}"
                        sto_modified = f"{' ' * (sto_initial_length - len(sto_modified))}{sto_modified}"

                        # place back in string and write for ownmax and sto_1
                        acct_line = acct_line[:column_widths["ownname"]] + ownmax_modified + sto_modified + acct_line[column_widths["sto_1"]:]
                        content.append(acct_line)

                    account_capture = False

                    capture = False

                    # append environmental line
                    content.append(line)

                else:
                    content.append(line)
                    capture = False

    # write modified output file
    template_file_base = os.path.splitext(os.path.basename(template_file))[0]
    output_file = os.path.join(output_dir, f"{template_file_base}_scenario-{scenario}_sample-{sample_id}.res")

    with open(output_file, "w") as out:
        for item in content:
            out.write(item)


def modify_res(output_dir: str,
               scenario: str,
               basin_name: str = "Gunnison",
               template_file: Union[None, str] = None,
               data_specification_file: Union[None, str] = None,
               target_structure_id_list: Union[None, List[str]] = None,
               skip_rows: int = 0,
               seed_value: Union[None, int] = None,
               n_jobs: int = -1,
               n_samples: int = 1):
    """Modify a single reservoir (.res) file based on a user provided sample.

    :param output_dir:          Path to output directory.
    :type output_dir:           str

    :param scenario:            Scenario name.
    :type scenario:             str

    :param sample:              An array of samples for each parameter.
    :type sample:               np.array

    :param sample_id:           Numeric ID of sample that is being processed. Defaults to 0.
    :type sample_id:            int

    :param basin_name:                      Name of basin for either:
                                                Upper_Colorado
                                                Yampa
                                                San_Juan
                                                Gunnison
                                                White
    :type basin_name:                       str

    :param template_file:       If a full path to a template file is provided it will be used.  Otherwise the
                                default template in this package will be used.
    :type template_file:        Union[None, str]

    :param data_specification_file:     If a full path to a data specification template is provided it will be used.
                                        Otherwise, the default file in the package is used.
    :type data_specification_file:      Union[None, str]

    :param target_structure_id_list:    Structure id list to process.  If None, all structure ids will be processed.
    :type target_structure_id_list:     Union[None, List[str]]

    :param skip_rows:           Number of rows to skip after the commented fields end; default 1
    :type skip_rows:            int, optional

    :param seed_value:                      Integer to use for random seed or None if not desired
    :type seed_value:                       Union[None, int] = None)

    :param n_jobs:                          Number of jobs to process in parallel.  Defaults to -1 meaning
                                            all but 1 processor.
    :type n_jobs:                           int

    :param n_samples:                       Used if generate_samples is True.  Number of samples to generate.
    :type n_samples:                        int

    :example:

    .. code-block:: python

        import statemodify as stm


        output_directory = "<your desired output directory>"
        scenario = "<your scenario name>"

        # basin name to process
        basin_name = "Gunnison"

        # seed value for reproducibility if so desired
        seed_value = 0

        # number of jobs to launch in parallel; -1 is all but 1 processor used
        n_jobs = 2

        # number of samples to generate
        n_samples = 2

        stm.modify_res(output_dir=output_directory,
                       scenario=scenario,
                       basin_name=basin_name,
                       target_structure_id_list=None,
                       seed_value=seed_value,
                       n_jobs=n_jobs,
                       n_samples=n_samples)

    """

    yaml_file = pkg_resources.resource_filename("statemodify", "data/parameter_definitions.yml")
    param_dict = utx.yaml_to_dict(yaml_file)

    # generate an array of samples to process
    sample_array = sampler.generate_sample_all_params(n_samples=n_samples,
                                                      seed_value=seed_value)

    # generate all files in parallel
    results = Parallel(n_jobs=n_jobs, backend="loky")(delayed(modify_single_res)(output_dir=output_dir,
                                                                                 scenario=scenario,
                                                                                 basin_name=basin_name,
                                                                                 sample=sample[param_dict["rstorage"]["index"]],
                                                                                 sample_id=sample_id,
                                                                                 template_file=template_file,
                                                                                 data_specification_file=data_specification_file,
                                                                                 target_structure_id_list=target_structure_id_list,
                                                                                 skip_rows=skip_rows) for sample_id, sample in enumerate(sample_array))

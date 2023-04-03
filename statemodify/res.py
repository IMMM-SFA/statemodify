import os
from typing import Union, List

import numpy as np

import statemodify.utils as utx


def get_reservoir_structure_ids(template_file: Union[None, str] = None,
                                data_specification_file: Union[None, str] = None,
                                column_width: int = 12):
    """Generate a list of structure ids that are in the input file.

    :param template_file:       If a full path to a template file is provided it will be used.  Otherwise the
                                default template in this package will be used.
    :type template_file:        Union[None, str]

    :param data_specification_file:     If a full path to a data specification template is provided it will be used.
                                        Otherwise, the default file in the package is used.
    :type data_specification_file:      Union[None, str]

    :param column_width:                Column width to search for at the beginning of each line for the structure id.
    :type column_width:                 int

    :returns:                           List of structure ids
    :rtype:                             List

    """

    # select the appropriate template file
    template_file = utx.select_template_file(template_file, extension="res")

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
                x = line[0:column_width]
                if len(x) > 0:
                    content.append(x)

    return content


def modify_single_res(output_dir: str,
                      scenario: str,
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
    template_file = utx.select_template_file(template_file, extension="res")

    # read in data specification yaml
    data_specification_file = utx.select_data_specification_file(yaml_file=data_specification_file,
                                                                 extension="res")
    data_spec_dict = utx.yaml_to_dict(data_specification_file)

    column_widths = data_spec_dict["column_widths"]

    if target_structure_id_list is None:
        target_structure_id_list = get_reservoir_structure_ids(template_file=template_file,
                                                               data_specification_file=data_specification_file)

    content = []
    with open(template_file) as get:
        for index, line in enumerate(get):

            # capture header
            if line[0] == data_spec_dict["comment_indicator"] or skip_rows != 0:
                content.append(line)

            # account for any additional non-commented lines that need to be skipped before starting data collection
            elif skip_rows > 0:
                content.append(line)
                skip_rows -= 1

            else:

                # get segment of line that usually houses structure id
                structure_id_segement = line[0:column_widths["id"]].strip()

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
                        # get original values
                        ownmax_initial = acct_line[column_widths["ownname"]:column_widths["ownmax"]]
                        ownmax_initial_length = len(ownmax_initial)

                        # break out account volume by total number of accounts in equal shares
                        ownmax_modified = f"{int(float(ownmax_initial) * sample)}."

                        # reapply space padding
                        n_space_padding = ownmax_initial_length - len(ownmax_modified)
                        ownmax_modified = f"{' ' * n_space_padding}{ownmax_modified}"

                        # place back in string and write for ownmax and sto_1
                        acct_line = acct_line[:column_widths["ownname"]] + ownmax_modified + ownmax_modified + acct_line[column_widths["sto_1"]:]
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



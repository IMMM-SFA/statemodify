import os
import pkg_resources
from typing import Union, List

import numpy as np
import pandas as pd

import statemodify.utils as utx


def modify_single_res(target_structure_id_list: List[str],
                      output_dir: str,
                      scenario: str,
                      sample: np.array,
                      sample_id: int = 0,
                      template_file: Union[None, str] = None,
                      data_specification_file: Union[None, str] = None,
                      skip_rows: int = 0,
                      terminate_string: str = "Evaporation"):

    # select the appropriate template file
    template_file = utx.select_template_file(template_file, extension="res")

    # read in data specification yaml
    data_specification_file = utx.select_data_specification_file(yaml_file=data_specification_file,
                                                                 extension="res")
    data_spec_dict = utx.yaml_to_dict(data_specification_file)

    column_widths = data_spec_dict["column_widths"]

    content = []
    with open(template_file) as get:
        for index, line in enumerate(get):

            # capture header
            if line[0] == data_spec_dict["comment_indicator"] or skip_rows != 0:
                content.append(line)

            # account for any additional non-commented lines that need to be skipped before starting data collection
            elif skip_rows >0:
                content.append(line)
                skip_rows -= 1

            else:

                # get segment of line that usually houses structure id
                structure_id_segement = line[0:column_widths["id"]].strip()

                # if reservoir start line found trigger capture
                if (len(structure_id_segement) > 0) and (capture is False) and (terminate_string not in line):

                    # keep content
                    content.append(line)

                    # set to capture next lines
                    capture = True

                    # assign structure id for current reservoir
                    structure_id = structure_id_segement

                # if capturing line
                elif (len(structure_id_segement) == 0) and (capture is True) and (terminate_string not in line) and (
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

                elif (len(structure_id_segement) == 0) and (capture is True) and (terminate_string in line) and (
                        structure_id in target_structure_id_list):

                    # compile accounts
                    for acct_line in account_list:
                        # get original values
                        ownmax_initial = acct_line[24:32]
                        ownmax_initial_length = len(ownmax_initial)

                        # break out account volume by total number of accounts in equal shares
                        ownmax_modified = f"{int(float(ownmax_initial) * sample)}."

                        # reapply space padding
                        n_space_padding = ownmax_initial_length - len(ownmax_modified)
                        ownmax_modified = f"{' ' * n_space_padding}{ownmax_modified}"

                        # place back in string and write for ownmax and rsto
                        acct_line = acct_line[:24] + ownmax_modified + ownmax_modified + acct_line[40:]
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



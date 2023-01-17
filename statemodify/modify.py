import os
import pkg_resources
from dataclasses import dataclass
from typing import Union, Dict, List

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import statemodify.utils as utx
import statemodify.sampler as sampler


def set_alignment(value: str,
                  n_spaces: int = 0,
                  align: str = "left") -> str:
    """Set left or right alignment.

    :param value:                   Value to evaluate.
    :type value:                    str

    :param n_spaces:                Number of spaces to buffer the value by. If less than the length
                                    of the value, no spaces will be added to the padding.
    :type n_spaces:                 int

    :param align:                   Either 'left' or 'right' alignment for the value.
    :type align:                    str

    :return:                        Value with string padding.

    """

    # set align attribute to lower case
    lowercase_align = align.casefold().strip()

    if lowercase_align == "left":
        return f"{value}{n_spaces * ' '}"

    elif lowercase_align == "right":
        return f"{n_spaces * ' '}{value}"

    else:
        raise AssertionError(f"Choice for alignment '{align}' not supported.  Must be 'left' or 'right'.")


def pad_with_spaces(value: str,
                    expected_width: int,
                    align="left") -> str:
    """Pad a string with the number of spaces specified by the user.

    :param value:                   Value to evaluate.
    :type value:                    str

    :param expected_width:          Expected width of the field.
    :type expected_width:           int

    :param align:                   Either 'left' or 'right' alignment for the value.
    :type align:                    str

    :return:                        Value with string padding.

    """

    # strip all whitespace padding from value
    value_stripped = value.strip()

    # get length of data in field
    field_length = len(value_stripped)

    if field_length <= expected_width:

        # get the number of missing spaces
        missing_spaces = expected_width - field_length

        return set_alignment(value=value_stripped,
                             n_spaces=missing_spaces,
                             align=align)

    else:

        # reduce precision to fit field if float
        try:

            split_value = value_stripped.split(".")

            if len(split_value) == 1:
                raise f"Column width '{field_length}' for value '{split_value[0]}' exceeds the expected width '{expected_width}'"

            # precision of value
            n_decimals = len(split_value[-1])

            # number of decimals overflowing
            overflow = n_decimals - (field_length - expected_width)

            # round to fit
            return np.float64(value_stripped).round(overflow).astype(str)

        except TypeError:
            raise f"Column width '{field_length}' for value '{value_stripped}' exceeds the expected width '{expected_width}'"


def add_zero_padding(x: str,
                     precision: int = 2) -> str:
    """Some fields expect zero padding that gets rounded off by pandas.
    This method adds that back in.

    :param x:                       Float value from file that is represented as a string.
    :type x:                        str

    :param precision:               Precision to account for.
    :type precision:                int

    :return:                        Zero padded string.

    """

    # get length of precision
    x_length = len(x.split(".")[-1])

    if x_length < precision:

        # determine the number of zeros needed
        n_zeros = precision - x_length

        return f"{x}{'0' * n_zeros}"
    else:
        return x


def populate_dict(line: str,
                  field_dict: dict,
                  column_widths: dict,
                  column_list: list,
                  data_types: dict) -> dict:
    """Populate the input dictionary with values from each line based on column widths.

    :param line:                    Line of data as a string from the input file.
    :type line:                     str

    :param field_dict:              Dictionary holding values for each field.
    :type field_dict:               dict

    :param column_widths:           Dictionary of column names to expected widths.
    :type column_widths:            dict

    :param column_list:             List of columns to process.
    :type column_list:              list

    :param data_types:              Dictionary of column names to data types.
    :type data_types:               dict

    :return:                        Populated data dictionary.

    """

    start_index = 0
    for idx, i in enumerate(column_list):

        if idx == 0:
            end_index = column_widths[i]

        else:
            end_index = start_index + column_widths[i]

        # extract portion of the line based on the known column width
        string_extraction = line[start_index: end_index]

        # convert to desired data type
        out_string = data_types[i](string_extraction)

        # append to dict
        field_dict[i].append(out_string)

        # advance start index for next iteration
        start_index += column_widths[i]

    return field_dict


def prep_data(field_dict: dict,
              template_file: str,
              column_list: list,
              column_widths: dict,
              data_types: dict,
              comment: str = "#",
              skip_rows: int = 0):
    """Ingest statemod file and format into a data frame.

    :param field_dict:              Dictionary holding values for each field.
    :type field_dict:               dict

    :param template_file:           Statemod input file to parse.
    :type template_file:            str

    :param column_widths:           Dictionary of column names to expected widths.
    :type column_widths:            dict

    :param column_list:             List of columns to process.
    :type column_list:              list

    :param data_types:              Dictionary of column names to data types.
    :type data_types:               dict

    :param comment:                 Characters leading string indicating ignoring a line.
    :type comment:                  str

    :param skip_rows:               The number of uncommented rows of data to skip.
    :type skip_rows:                int

    :return:                        [0] data frame of data from file
                                    [1] header data from file

    """

    # empty string to hold header data
    header = ""

    capture = False
    with open(template_file) as template:

        for idx, line in enumerate(template):

            if capture:

                # populate dictionary with data content
                field_dict = populate_dict(line, field_dict, column_widths, column_list, data_types)

            else:

                # passes all commented lines in header
                if line[0] != comment:

                    # if you are not skipping any rows that are not comments
                    if skip_rows == 0:

                        field_dict = populate_dict(line, field_dict, column_widths, column_list, data_types)
                        capture = True

                    else:

                        # count down the number of rows to skip
                        skip_rows -= 1

                        # store any header and preliminary lines to use in restoration
                        header += line

                else:
                    header += line

    # convert dictionary to a pandas data frame
    df = pd.DataFrame(field_dict)

    return df, header


def construct_outfile_name(template_file: str,
                           output_directory: str,
                           scenario: str,
                           sample_id: int) -> str:
    """Construct output file name from input template.

    :param template_file:           Statemod input file to parse.
    :type template_file:            str

    :param output_directory:        Output directory to save outputs to.
    :type output_directory:         str

    :param scenario:                Scenario name.
    :type scenario:                 str

    :param sample_id:               ID of sample.
    :type sample_id:                int

    :return:                        Full path with file name and extension for the modified output file.

    """

    # extract file basename
    template_basename = os.path.basename(template_file)

    # split basename into filename and extension
    template_name_parts = os.path.splitext(template_basename)

    # output file name
    output_file = f"{template_name_parts[0]}_scenario-{scenario}_sample-{sample_id}{template_name_parts[-1]}"

    return os.path.join(output_directory, output_file)


def construct_data_string(df: pd.DataFrame,
                          column_list: list,
                          column_widths: dict,
                          column_alignment: dict) -> str:
    """Format line and construct data string.

    :param df:                      ID of sample.
    :type df:                       pd.DataFrame

    :param column_widths:           Dictionary of column names to expected widths.
    :type column_widths:            dict

    :param column_list:             List of columns to process.
    :type column_list:              list

    :param column_alignment:        Dictionary of column names to their expected alignment (e.g., right, left).
    :type column_alignment:         dict

    :return:                        Formatted data string.

    """

    # initialize empty string to store data in
    data = ""

    # for each row of data
    for idx in df.index:

        # for each column in row
        for i in column_list:

            # add to output string
            data += pad_with_spaces(df[i][idx],
                                    column_widths[i],
                                    align=column_alignment[i])
        # add in newline
        data += "\n"

    return data


def apply_adjustment_factor(data_df: pd.DataFrame,
                            value_columns: list,
                            query_field: str,
                            target_ids: list,
                            factor: float,
                            factor_method: str = "add") -> pd.DataFrame:
    """Apply adjustment to template file values for target ids using a sample factor.

    :param data_df:                         Data frame of data content from file.
    :type data_df:                          pd.DataFrame

    :param value_columns:                   Value columns that may be modified.
    :type value_columns:                    list

    :param query_field:                     Field name to conduct queries for.
    :type query_field:                      str

    :param target_ids:                      Ids associated in query field to modify.
    :type target_ids:                       list

    :param factor:                          Value to multiply the selected value columns by.
    :type factor:                           float

    :param factor_method:                   Method by which to apply the factor. Options 'add', 'multiply'.
                                            Defaults to 'add'.
    :type factor_method:                    str

    :return:                                An data frame of modified values to replace the original data with.
    :rtype:                                 pd.DataFrame

    """

    if factor_method == "add":
        return (data_df[value_columns] + factor).where(data_df[query_field].isin(target_ids), data_df[value_columns])

    elif factor_method == "multiply":
        return (data_df[value_columns] * factor).where(data_df[query_field].isin(target_ids), data_df[value_columns])

    else:
        raise KeyError(f"'factor_method' value {factor_method} is not in available options of ('add', 'multiply').")


def validate_bounds(bounds_list: List[List[float]],
                    min_value: float = -0.5,
                    max_value: float = 1.0) -> None:
    """Ensure sample bounds provided by the user conform to a feasible range of values in feet per month.

    :param bounds_list:             List of bounds to use for each parameter.
    :type bounds_list:              List[List[float]]

    :param min_value:               Minimum value of bounds that is feasible. Default -0.5.
    :type max_value:                float

    :param max_value:               Maximum value of bounds that is feasible. Default 1.0.
    :type max_value:                float

    :return:                        None
    :rtype:                         None

    :example:

        .. code-block:: python

        import statemodify as stm

        # list of bounds to use for each parameter
        bounds_list = [[-0.5, 1.0], [-0.5, 1.0]]

        # validate bounds
        stm.validate_bounds(bounds_list=bounds_list,
                            min_value=-0.5,
                            max_value=1.0)

    """

    flag_min_error = False
    flag_max_error = False

    # get min of lists
    l_min = min([min(i) for i in bounds_list])

    # get max of lists
    l_max = max([max(i) for i in bounds_list])

    if l_min < min_value:
        msg_min = f"Minimum value for feasible sample bound '{l_min}' is invalid.  Minimum possible bound is '{min_value}'."
        flag_min_error = True

    if l_max > max_value:
        msg_max = f"Maximum value for feasible sample bound '{l_max}' is invalid.  Maximum possible bound is '{max_value}'."
        flag_max_error = True

    if flag_min_error and flag_max_error:
        raise ValueError((msg_min, msg_max))

    elif flag_min_error and flag_max_error is False:
        raise ValueError(msg_min)

    elif flag_min_error is False and flag_max_error:
        raise ValueError(msg_max)

    else:
        return None




@dataclass
class Modify:
    """Modification template for data transformation.

    :param query_field:         Field name to use for target query.
    :type query_field:          str

    :param skip_rows:           Number of rows to skip after the commented fields end
    :type skip_rows:            int, optional

    :param min_bound_value:     Minimum value allowable for the sample generator to produce.
    :type min_bound_value:      float

    :param max_bound_value:     Maximum value allowable for the sample generator to produce.
    :type max_bound_value:      float

    :param comment_indicator:   Character(s) indicating the line in the file is a comment. Defualt "#"
    :type comment_indicator:    str

    :param data_dict:           Field specification in the file as a dictionary to hold the values for each field
    :type data_dict:            Dict[str, List[None]]

    :param column_widths:       Column widths for the output file as a dictionary.
    :type column_widths:        Dict[str, int]

    :param column_alignment:    Expected column alignment.
    :type column_alignment:     Dict[str, str]

    :param data_types:          Expected data types for each field
    :type data_types:           Dict[str, type]

    :param column_list:         List of columns to process.
    :type column_list:          List[str]

    :param value_columns:       List of columns that may be modified.
    :type value_columns:        List[str]

    """

    query_field: str
    skip_rows: int
    min_bound_value: float
    max_bound_value: float
    comment_indicator: str
    data_dict: Dict[str, List[None]]
    column_widths: Dict[str, int]
    column_alignment: Dict[str, str]
    data_types: Dict[str, type]
    column_list: List[str]
    value_columns: List[str]



class ModifyTemplate:
    """Modification template for data transformation."""

    # set minimum and maximum feasible sampling bounds in feet per month
    MIN_BOUND_VALUE: float = -0.5
    MAX_BOUND_VALUE: float = 1.0

    def __init__(self,
                 query_field: str,
                 skip_rows: int = 1,
                 template_file: Union[None, str] = None):

        """Data specifics for the StateMod evapotranspiration files (.eva) file specification.


        :param query_field:         Field name to use for target query.
        :type query_field:          str

        :param skip_rows:           Number of rows to skip after the commented fields end; default 1
        :type skip_rows:            int, optional

        :param template_file:       If a full path to a template file is provided it will be used.  Otherwise the
                                    default template in this package will be used.
        :type template_file:        Union[None, str]

        """

        self.query_field = query_field
        self.skip_rows = skip_rows

        # character indicating row is a comment
        self.comment = "#"

        # dictionary to hold values for each field
        self.data_dict = {"prefix": [],
                          "id": [],
                          "oct": [],
                          "nov": [],
                          "dec": [],
                          "jan": [],
                          "feb": [],
                          "mar": [],
                          "apr": [],
                          "may": [],
                          "jun": [],
                          "jul": [],
                          "aug": [],
                          "sep": []}

        # define the column widths for the output file
        self.column_widths = {"prefix": 5,
                              "id": 12,
                              "oct": 8,
                              "nov": 8,
                              "dec": 8,
                              "jan": 8,
                              "feb": 8,
                              "mar": 8,
                              "apr": 8,
                              "may": 8,
                              "jun": 8,
                              "jul": 8,
                              "aug": 8,
                              "sep": 8}

        # expected column alignment
        self.column_alignment = {"prefix": "left",
                                  "id": "left",
                                  "oct": "right",
                                  "nov": "right",
                                  "dec": "right",
                                  "jan": "right",
                                  "feb": "right",
                                  "mar": "right",
                                  "apr": "right",
                                  "may": "right",
                                  "jun": "right",
                                  "jul": "right",
                                  "aug": "right",
                                  "sep": "right"}

        # expected data types for each field
        self.data_types = {"prefix": str,
                           "id": str,
                           "oct": np.float64,
                           "nov": np.float64,
                           "dec": np.float64,
                           "jan": np.float64,
                           "feb": np.float64,
                           "mar": np.float64,
                           "apr": np.float64,
                           "may": np.float64,
                           "jun": np.float64,
                           "jul": np.float64,
                           "aug": np.float64,
                           "sep": np.float64}

        # list of columns to process
        self.column_list = ["prefix", "id", "oct", "nov", "dec", "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep"]

        # list of value columns that may be modified
        self.value_columns = ["oct", "nov", "dec", "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep"]

        # template file selection
        if template_file is None:
            self.template_file = pkg_resources.resource_filename("statemodify", "data/template.eva")
        else:
            self.template_file = template_file

        # prepare template data frame for alteration
        self.template_df, self.template_header = utx.prep_data(field_dict=self.data_dict,
                                                               template_file=self.template_file,
                                                               column_list=self.column_list,
                                                               column_widths=self.column_widths,
                                                               data_types=self.data_types,
                                                               comment=self.comment,
                                                               skip_rows=self.skip_rows)


def modify_single_eva(modify_dict: Dict[str, List[Union[str, float]]],
                      query_field: str,
                      output_dir: str,
                      scenario: str,
                      sample: np.array,
                      sample_id: int = 0,
                      skip_rows: int = 1,
                      template_file: Union[None, str] = None,
                      factor_method: str = "add") -> None:
    """Modify StateMod net reservoir evaporation annual data file (.eva) using a Latin Hypercube Sample from the user.
    Samples are processed in parallel. Modification is targeted at 'municipal' and 'standard' fields where ids to
    modify are specified in the `modify_dict` argument.  The user must specify bounds for each field name.

    :param modify_dict:         Dictionary of parameters to setup the sampler.  See following example.
    :type modify_dict:          Dict[str, List[Union[str, float]]]

    :param query_field:         Field name to use for target query.
    :type query_field:          str

    :param sample:              An array of samples for each parameter.
    :type sample:               np.array

    :param sample_id:           Numeric ID of sample that is being processed. Defaults to 0.
    :type sample_id:            int

    :param output_dir:          Path to output directory.
    :type output_dir:           str

    :param scenario:            Scenario name.
    :type scenario:             str

    :param skip_rows:           Number of rows to skip after the commented fields end; default 1
    :type skip_rows:            int, optional

    :param template_file:       If a full path to a template file is provided it will be used.  Otherwise the
                                default template in this package will be used.
    :type template_file:        Union[None, str]

    :param factor_method:       Method by which to apply the factor. Options 'add', 'multiply'.
                                Defaults to 'add'.
    :type factor_method:        str

    :return: None
    :rtype: None

    :example:

    .. code-block:: python

        import statemodify as stm

        # a dictionary to describe what you want to modify and the bounds for the LHS
        setup_dict = {
            "ids": [["10001", "10004"], ["10005", "10006"]],
            "bounds": [[-0.5, 1.0], [-0.5, 1.0]]
        }

        output_directory = "<your desired output directory>"
        scenario = "<your scenario name>"

        # sample id for the current run
        sample_id = 0

        # sample array for each parameter
        sample = np.array([0.39, -0.42])

        # seed value for reproducibility if so desired
        seed_value = None

        # number of rows to skip in file after comment
        skip_rows = 1

        # name of field to query
        query_field = "id"

        # number of jobs to launch in parallel; -1 is all but 1 processor used
        n_jobs = -1

        # generate a batch of files using generated LHS
        stm.modify_single_eva(modify_dict=modify_dict,
                              query_field=query_field,
                              sample=sample,
                              sample_id=sample_id,
                              output_dir=output_dir,
                              scenario=scenario,
                              skip_rows=skip_rows,
                              template_file=None,
                              factor_method="add")

    """

    # instantiate specification class
    mod = ModifyEva(query_field=query_field,
                    skip_rows=skip_rows,
                    template_file=template_file)

    # strip the query field of any whitespace
    mod.template_df[mod.query_field] = mod.template_df[mod.query_field].str.strip()

    # validate user provided sample bounds to ensure they are within a feasible range
    utx.validate_bounds(bounds_list=modify_dict["bounds"],
                        min_value=mod.MIN_BOUND_VALUE,
                        max_value=mod.MAX_BOUND_VALUE)

    # modify value columns associated structures based on the sample draw
    for index, i in enumerate(modify_dict["names"]):

        # extract target ids to modify
        id_list = modify_dict["ids"][index]

        # extract factors from sample for the subset and sample
        factor = sample[index]

        # apply adjustment
        mod.template_df[mod.value_columns] = utx.apply_adjustment_factor(data_df=mod.template_df,
                                                                         value_columns=mod.value_columns,
                                                                         query_field=mod.query_field,
                                                                         target_ids=id_list,
                                                                         factor=factor,
                                                                         factor_method=factor_method)

    # reconstruct precision
    mod.template_df[mod.value_columns] = mod.template_df[mod.value_columns].round(4)

    # convert all fields to str type
    mod.template_df = mod.template_df.astype(str)

    # add formatted data to output string
    data = utx.construct_data_string(mod.template_df, mod.column_list, mod.column_widths, mod.column_alignment)

    # write output file
    output_file = utx.construct_outfile_name(mod.template_file, output_dir, scenario, sample_id)

    with open(output_file, "w") as out:
        # write header
        out.write(mod.template_header)

        # write data
        out.write(data)


def modify_eva(modify_dict: Dict[str, List[Union[str, float]]],
               query_field: str,
               output_dir: str,
               scenario: str,
               sampling_method: str = "LHS",
               n_samples: int = 1,
               skip_rows: int = 1,
               n_jobs: int = -1,
               seed_value: Union[None, int] = None,
               template_file: Union[None, str] = None,
               factor_method: str = "add") -> None:
    """Modify StateMod net reservoir evaporation annual data file (.eva) using a Latin Hypercube Sample from the user.
    Samples are processed in parallel. Modification is targeted at 'municipal' and 'standard' fields where ids to
    modify are specified in the `modify_dict` argument.  The user must specify bounds for each field name.

    :param modify_dict:         Dictionary of parameters to setup the sampler.  See following example.
    :type modify_dict:          Dict[str, List[Union[str, float]]]

    :param query_field:         Field name to use for target query.
    :type query_field:          str

    :param output_dir:          Path to output directory.
    :type output_dir:           str

    :param scenario:            Scenario name.
    :type scenario:             str

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

    :return: None
    :rtype: None

    :example:

    .. code-block:: python

        import statemodify as stm

        # a dictionary to describe what you want to modify and the bounds for the LHS
        setup_dict = {
            "ids": [["10001", "10004"], ["10005", "10006"]],
            "bounds": [[-0.5, 1.0], [-0.5, 1.0]]
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

        # generate a batch of files using generated LHS
        stm.modify_eva(modify_dict=modify_dict,
                       query_field=query_field,
                       output_dir=output_dir,
                       scenario=scenario,
                       sampling_method="LHS",
                       n_samples=n_samples,
                       skip_rows=skip_rows,
                       n_jobs=n_jobs,
                       seed_value=seed_value,
                       template_file=None,
                       factor_method="add")

    """

    # build a problem dictionary for use by SALib
    problem_dict = sampler.build_problem_dict(modify_dict, fill=True)

    # generate a sample array
    sample_array = sampler.generate_samples(problem_dict=problem_dict,
                                            n_samples=n_samples,
                                            sampling_method=sampling_method,
                                            seed_value=seed_value)

    # generate all files in parallel
    results = Parallel(n_jobs=n_jobs, backend="loky")(delayed(modify_single_eva)(modify_dict=modify_dict,
                                                                                 query_field=query_field,
                                                                                 sample=sample,
                                                                                 sample_id=sample_id,
                                                                                 output_dir=output_dir,
                                                                                 scenario=scenario,
                                                                                 skip_rows=skip_rows,
                                                                                 template_file=template_file,
                                                                                 factor_method=factor_method) for sample_id, sample in enumerate(sample_array))

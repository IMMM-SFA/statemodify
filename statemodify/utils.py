import pandas as pd


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
        raise AssertionError(f"Column width '{field_length}' exceeds the expected width '{expected_width}'")


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

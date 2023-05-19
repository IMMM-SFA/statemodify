import os
from typing import Union

import pandas as pd

import statemodify.utils as utx


def extract_xre_data(structure_name: str,
                     structure_id: str,
                     input_file: Union[None, str] = None,
                     basin_name: Union[None, str] = None,
                     write_csv: bool = False,
                     write_parquet: bool = False,
                     output_directory: Union[str, None] = None) -> pd.DataFrame:
    """Extract a single reservoir from a raw .xre file in to a Pandas data frame.
    Optionally save output to a CSV or Parquet file.

    :param structure_name: str, name of the reservoir
    :param structure_id: str, structure ID for reservoir of interest
    :param input_file: Union[None, str], path to the xre file
    :param basin_name: Union[None, str], Name of basin for either:
                                                Upper_Colorado
                                                Yampa
                                                San_Juan
                                                Gunnison
                                                White
    :param write_csv: bool, whether to write output to a CSV file
    :param write_parquet: bool, whether to write output to a Parquet file
    :param output_directory: Union[str, None], path to the output directory

    :return: pd.DataFrame, a Pandas data frame containing the extracted reservoir data

    :example:

    .. code-block::python

        xre_file = '<path_to_file>/gm2015H.xre'  # path the the xre file
        structure_ID = '2803590'  # structure ID for reservoir of interest
        structure_name = 'Blue_Mesa'  # name of the reservoir

        df = extract_xre_data(structure_name=structure_name,
                              structure_id=structure_ID,
                              input_file=xre_file,
                              output_directory=None,
                              write_csv=None,
                              write_parquet=None
        )

    """

    # select the appropriate input file if none provided
    if input_file is None and basin_name is None:
        raise ValueError(f"Either input_file or basin_name must be set.  If using template, set basin_name.")
    elif input_file is None and basin_name is not None:
        input_file = utx.select_template_file(basin_name, input_file, extension="xre")

    # Open the .res file and grab the contents of the file
    with open(input_file, 'r') as f:
        all_data_xre = [x for x in f.readlines()]

    with open(input_file, 'r') as f:
        all_split_data_xre = [x.split('.') for x in f.readlines()]

    out_data = [['Res ID', 'ACC', 'Year', 'MO', 'Init. Storage', 'From River By Priority', 'From River By Storage',
                 'From River By Other', 'From River By Loss', 'From Carrier By Priority',
                 'From Carrier By Other', 'From Carrier By Loss', 'Total Supply', 'From Storage to River For Use',
                 'From Storage to River for Exc', 'From Storage to Carrier for use', 'Total Release', 'Evap',
                 'Seep and Spill', 'EOM Content', 'Target Stor', 'Decree Lim', 'River Inflow', 'River Release',
                 'River Divert',
                 'River by Well', 'River Outflow']]

    # ensure an output directory is set if writing a file
    if (write_csv or write_parquet) and (output_directory is None):
        raise ValueError("Please set the output_directory if writing a file.")

    # loop through each line and identify the structure of interest
    for i in range(len(all_data_xre)):
        row_id_data = []  # this will store the structure ID, account num, year, month and init storage
        row_id_data.extend(all_split_data_xre[i][0].split())  # loading the data described above
        if row_id_data and row_id_data[0] == structure_id:  # if the structure is the one we're looking for
            row_detail_data = all_split_data_xre[i][1:]  # first grab the initial data
            out_data.append(
                row_id_data + row_detail_data)  # combine it with the rest of the data and append to out_data

    # reformat to dictionary
    data_dict = {}
    for index, row in enumerate(out_data):

        # capture header
        if index == 0:
            column_names = []
            for i in row:
                x = i.strip()
                column_names.append(x)
                data_dict[x] = []

        else:
            for ix, i in enumerate(column_names):
                x = row[ix].strip()
                data_dict[i].append(x)

    # convert to data frame
    df = pd.DataFrame(data_dict)

    if write_csv:
        output_csv = os.path.join(output_directory, f"{structure_name}_xre_data.csv")
        df.to_csv(output_csv, index=False)

    if write_parquet:
        output_parquet = os.path.join(output_directory, f"{structure_name}_xre_data.parquet")
        df.to_parquet(output_parquet)

    return df

import pkg_resources
from typing import Union, Dict, List

import numpy as np
from joblib import Parallel, delayed
from SALib.sample import latin

import statemodify.utils as utx


class ModifyEva:
    """Modify .eva files."""

    def __init__(self,
                 modify_dict: Dict[str, List[Union[str, float]]],
                 query_field: str,
                 n_samples: int = 1,
                 skip_rows: int = 1,
                 seed_value: Union[None, int] = None):
        """Modifier for evapotranspiration files.


        :param modify_dict:         Dictionary of parameters to setup the sampler.
        :type modify_dict:          Dict[str, List[Union[str, float]]]

        :param query_field:         Field name to use for target query.
        :type query_field:          str

        :param n_samples:           Number of LHS samples to generate, optional. Defaults to 1.
        :type n_samples:            int, optional

        :param skip_rows:           Number of rows to skip after the commented fields end; default 1
        :type skip_rows:            int, optional

        :param seed_value: Seed value to use when generating samples for the purpose of reproducibility. Defaults to None.
        :type seed_value: Union[None, int], optional

        """

        self.modify_dict = modify_dict
        self.query_field = query_field
        self.n_samples = n_samples
        self.seed_value = seed_value
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

        # template file
        self.template_file = pkg_resources.resource_filename("statemodify", "data/template.eva")

        # build problem set
        self.problem = self.build_problem()

        # generate samples and add them to modify dict
        self.modify_dict["samples"] = self.generate_samples()

    def generate_data(self):
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

        return utx.prep_data(field_dict=self.data_dict,
                             template_file=self.template_file,
                             column_list=self.column_list,
                             column_widths=self.column_widths,
                             data_types=self.data_types,
                             comment=self.comment,
                             skip_rows=self.skip_rows)

    def validate_modify_dict(self):
        """Ensure modify dictionary is complete."""

        pass

    def build_problem(self):
        """Build the problem set from the input modify dictionary."""

        return {
            'num_vars': len(self.modify_dict["names"]),
            'names': self.modify_dict["names"],
            'bounds': self.modify_dict["bounds"]
        }

    def generate_samples(self):
        """Generate samples."""

        # generate our sample
        return latin.sample(problem=self.problem,
                            N=self.n_samples,
                            seed=self.seed_value).T


def modify_single_eva(modify_dict: Dict[str, List[Union[str, float]]],
                      query_field: str,
                      sample_id: int,
                      output_dir: str,
                      scenario: str,
                      n_samples: int = 1,
                      skip_rows: int = 1,
                      seed_value: Union[None, int] = None):

    # instantiate
    mod = ModifyEva(modify_dict=modify_dict,
                    query_field=query_field,
                    n_samples=n_samples,
                    skip_rows=skip_rows,
                    seed_value=seed_value)

    # copy template data frame for alteration
    df, header = mod.generate_data()

    # strip the query field of any whitespace
    df[mod.query_field] = df[mod.query_field].str.strip()

    # modify value columns associated structures based on the sample draw
    for index, i in enumerate(mod.modify_dict["names"]):

        # extract target ids to modify
        id_list = mod.modify_dict["ids"][index]

        # extract factors from sample for the subset and sample
        factor = mod.modify_dict["samples"][index][sample_id]

        df[mod.value_columns] = utx.apply_adjustment_factor(df, mod.value_columns, mod.query_field, id_list, factor)

    # reconstruct precision
    df[mod.value_columns] = df[mod.value_columns].round(4)

    # convert all fields to str type
    df = df.astype(str)

    # add formatted data to output string
    data = utx.construct_data_string(df, mod.column_list, mod.column_widths, mod.column_alignment)

    # write output file
    output_file = utx.construct_outfile_name(mod.template_file, output_dir, scenario, sample_id)

    with open(output_file, "w") as out:
        # write header
        out.write(header)

        # write data
        out.write(data)


def modify_eva(modify_dict: Dict[str, List[Union[str, float]]],
               query_field: str,
               output_dir: str,
               scenario: str,
               n_samples: int = 1,
               skip_rows: int = 1,
               n_jobs: int = -1,
               seed_value: Union[None, int] = None):
    """TODO: add the additional params
    Modify StateMod net reservoir evaporation annual data file (.eva) using a Latin Hypercube Sample from the user.
    Samples are processed in parallel. Modification is targeted at 'municipal' and 'standard' fields where ids to
    modify are specified in the `modify_dict` argument.  The user must specify bounds for each field name.

    :param modify_dict: Dictionary of parameters to setup the sampler.  See following example.
    :type modify_dict: Dict[str, List[Union[str, float]]]

    :param output_dir: Path to output directory.
    :type output_dir: str

    :param scenario: Scenario name.
    :type scenario: str

    :param n_samples: Number of LHS samples to generate, optional. Defaults to 1.
    :type n_samples: int, optional

    :param seed_value: Seed value to use when generating samples for the purpose of reproducibility. Defaults to None.
    :type seed_value: Union[None, int], optional

    :return: None
    :rtype: None

    :example:

    .. code-block:: python

        import statemodify as stm

        # a dictionary to describe what you want to modify and the bounds for the LHS
        setup_dict = {
            "names": ["municipal", "standard"],
            "ids": [["10001", "10004"], ["10005", "10006"]],
            "bounds": [[-1.0, 1.0], [-1.0, 1.0]]
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
                       sample_id=sample_id,
                       output_dir=output_dir,
                       scenario=scenario,
                       n_samples=n_samples,
                       skip_rows=skip_rows,
                       n_jobs=n_jobs,
                       seed_value=seed_value)

    """

    # generate all files in parallel
    results = Parallel(n_jobs=n_jobs, backend="loky")(delayed(modify_single_eva)(modify_dict=modify_dict,
                                                                                 query_field=query_field,
                                                                                 sample_id=sample_id,
                                                                                 output_dir=output_dir,
                                                                                 scenario=scenario,
                                                                                 n_samples=n_samples,
                                                                                 skip_rows=skip_rows,
                                                                                 seed_value=seed_value) for sample_id in range(n_samples))
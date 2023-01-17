import pkg_resources
from typing import Union, Dict, List

import numpy as np
from joblib import Parallel, delayed

import statemodify.utils as utx
import statemodify.sampler as sampler


class ModifyDdm:
    """Data specifics for the StateMod municipal, industrial, transbasin Demands (.ddm) file specification."""

    # set minimum and maximum feasible sampling bounds in feet per month
    MIN_BOUND_VALUE: float = 0.5
    MAX_BOUND_VALUE: float = 1.5

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
        self.data_dict = {"yr": [],
                          "break": [],
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
                          "sep": [],
                          "total": []}

        # define the column widths for the output file
        self.column_widths = {"yr": 4,
                              "break": 1,
                              "id": 12,
                              "jan": 8,
                              "feb": 8,
                              "mar": 8,
                              "apr": 8,
                              "may": 8,
                              "jun": 8,
                              "jul": 8,
                              "aug": 8,
                              "sep": 8,
                              "oct": 8,
                              "nov": 8,
                              "dec": 8,
                              "total": 10}

        # expected column alignment
        self.column_alignment = {"yr": "left",
                                 "break": "right",
                                 "id": "left",
                                 "jan": "right",
                                 "feb": "right",
                                 "mar": "right",
                                 "apr": "right",
                                 "may": "right",
                                 "jun": "right",
                                 "jul": "right",
                                 "aug": "right",
                                 "sep": "right",
                                 "oct": "right",
                                 "nov": "right",
                                 "dec": "right",
                                 "total": "right"}

        # expected data types for each field
        self.data_types = {"yr": int,
                           "break": str,
                           "id": str,
                           "jan": np.float64,
                           "feb": np.float64,
                           "mar": np.float64,
                           "apr": np.float64,
                           "may": np.float64,
                           "jun": np.float64,
                           "jul": np.float64,
                           "aug": np.float64,
                           "sep": np.float64,
                           "oct": np.float64,
                           "nov": np.float64,
                           "dec": np.float64,
                           "total": np.float64}

        # list of columns to process
        self.column_list = ["yr", "break", "id", "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec", "total"]

        # list of value columns that may be modified
        self.value_columns = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]

        # template file selection
        if template_file is None:
            self.template_file = pkg_resources.resource_filename("statemodify", "data/template.ddm")
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


def modify_single_ddm(modify_dict: Dict[str, List[Union[str, float]]],
                      query_field: str,
                      output_dir: str,
                      scenario: str,
                      sample: np.array,
                      sample_id: int = 0,
                      skip_rows: int = 1,
                      template_file: Union[None, str] = None,
                      factor_method: str = "multiply") -> None:
    """Modify StateMod municipal, industrial, transbasin Demands (.ddm) using a Latin Hypercube Sample from the user.
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
            "bounds": [[0.5, 1.5], [0.5, 1.5]]
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
        stm.modify_single_ddm(modify_dict=modify_dict,
                              query_field=query_field,
                              sample=sample,
                              sample_id=sample_id,
                              output_dir=output_dir,
                              scenario=scenario,
                              skip_rows=skip_rows,
                              template_file=None,
                              factor_method="multiply")

    """

    # instantiate specification class
    mod = ModifyDdm(query_field=query_field,
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


def modify_ddm(modify_dict: Dict[str, List[Union[str, float]]],
               output_dir: str,
               scenario: str,
               n_samples: int = 1,
               seed_value: Union[None, int] = None,
               template_file: str = pkg_resources.resource_filename("statemodify", "data/template.ddm"),
               query_field: str = "id") -> None:
    """Modify StateMod .ddm data file using a Latin Hypercube Sample from the user.  Samples are processed in parallel.
    Modification is targeted at 'municipal' and 'standard' fields where ids to modify are specified in the `modify_dict`
    argument.  The user must specify bounds for each field name.

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

    :param template_file: Path to the DDM template file, optional. Defaults to a DDM template included in the statemodify package.
    :type template_file: str, optional

    :param query_field: Field used for mapping parameters to DDM, optional. Defaults to "id".
    :type query_field: str, optional

    :return: None
    :rtype: None

    :example:

    .. code-block:: python

        import statemodify as stm

        # a dictionary to describe what you want to modify and the bounds for the LHS
        setup_dict = {
            "names": ["municipal", "standard"],
            "ids": [["7200764", "7200813CH"], ["7200764_I", "7200818"]],
            "bounds": [[-1.0, 1.0], [-1.0, 1.0]]
        }

        output_directory = "<your desired output directory>"
        scenario = "<your scenario name>"

        # the number of samples you wish to generate
        n_samples = 4

        # seed value for reproducibility if so desired
        seed_value = None

        # my template file.  If none passed into the `modify_ddm` function, the default file will be used.
        template_file = "<your ddm template file>"

        # the field that you want to use to query and modify the data
        query_field = "id"

        # generate a batch of files using generated LHS
        stm.modify_ddm(modify_dict=setup_dict,
                       output_dir=output_directory,
                       scenario=scenario,
                       n_samples=n_samples,
                       seed_value=seed_value,
                       query_field=query_field,
                       template_file=template_file)

    """

    # generate all files in parallel
    results = Parallel(n_jobs=-1, backend="loky")(delayed(modify_single_ddm)(modify_dict=modify_dict,
                                                                             sample_id=sample_id,
                                                                             output_dir=output_dir,
                                                                             scenario=scenario,
                                                                             n_samples=n_samples,
                                                                             seed_value=seed_value,
                                                                             template_file=template_file,
                                                                             query_field=query_field) for sample_id in range(n_samples))


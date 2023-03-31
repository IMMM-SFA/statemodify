import os
import pkg_resources as pkg
import random
from typing import List, Union

import numpy as np
import pandas as pd
from scipy import stats as ss
from hmmlearn import hmm
from joblib import Parallel, delayed
import statsmodels.api as sm

import statemodify.utils as utx
import statemodify.modify as modify
import statemodify.sampler as sampler


class GenerateIwrData:

    def __init__(self,
                 skip_rows: int = 1,
                 template_file: Union[None, str] = None,
                 data_specification_file: Union[None, str] = None):
        """Generate IWR data from user desired input template and file specification.  Defaults used if none provided.

        :param skip_rows:                                       Number of rows after comments to skip. Default 1.
        :type skip_rows:                                        int

        :param template_file:                                   File to use as template.  Default None.
        :type template_file:                                    Union[None, str]

        :param data_specification_file:                         Data specification YAML file.  Default None.
        :type data_specification_file:                          Union[None, str]

        :return template_file:                                  Selected template file path.
        :rtype template_file:                                   str

        :return data_specification_file:                        Selected data specification YAML file path.
        :rtype data_specification_file:                         str

        :return data_spec_dict:                                 Dictionary of data specification.
        :rtype data_spec_dict:                                  dict

        :return file_spec:                                      File modification specs.
        :rtype file_spec:                                       class

        :return template_df:                                    Processed data from file.
        :rtype template_df:                                     pd.DataFrame

        :return: template_header:                               Header for input template file.
        :rtype: template_header:                                list

        """

        # select the appropriate template file
        self.template_file = utx.select_template_file(template_file, extension="iwr")

        # read in data specification yaml
        self.data_specification_file = utx.select_data_specification_file(yaml_file=data_specification_file,
                                                                          extension="iwr")
        self.data_spec_dict = utx.yaml_to_dict(self.data_specification_file)

        # instantiate data specification and validation class
        self.file_spec = modify.Modify(comment_indicator=self.data_spec_dict["comment_indicator"],
                                       data_dict=self.data_spec_dict["data_dict"],
                                       column_widths=self.data_spec_dict["column_widths"],
                                       column_alignment=self.data_spec_dict["column_alignment"],
                                       data_types=self.data_spec_dict["data_types"],
                                       column_list=self.data_spec_dict["column_list"],
                                       value_columns=self.data_spec_dict["value_columns"])

        # prepare template data frame for alteration
        self.template_df, self.template_header = modify.prep_data(field_dict=self.file_spec.data_dict,
                                                                  template_file=self.template_file,
                                                                  column_list=self.file_spec.column_list,
                                                                  column_widths=self.file_spec.column_widths,
                                                                  data_types=self.file_spec.data_types,
                                                                  comment=self.file_spec.comment_indicator,
                                                                  skip_rows=skip_rows)


class GenerateXbmData:

    def __init__(self,
                 skip_rows: int = 1,
                 template_file: Union[None, str] = None,
                 data_specification_file: Union[None, str] = None):
        """Generate XBM data from user desired input template and file specification.  Defaults used if none provided.

        :param skip_rows:                                       Number of rows after comments to skip. Default 1.
        :type skip_rows:                                        int

        :param template_file:                                   File to use as template.  Default None.
        :type template_file:                                    Union[None, str]

        :param data_specification_file:                         Data specification YAML file.  Default None.
        :type data_specification_file:                          Union[None, str]

        :return template_file:                                  Selected template file path.
        :rtype template_file:                                   str

        :return data_specification_file:                        Selected data specification YAML file path.
        :rtype data_specification_file:                         str

        :return data_spec_dict:                                 Dictionary of data specification.
        :rtype data_spec_dict:                                  dict

        :return file_spec:                                      File modification specs.
        :rtype file_spec:                                       class

        :return template_df:                                    Processed data from file.
        :rtype template_df:                                     pd.DataFrame

        :return: template_header:                               Header for input template file.
        :rtype: template_header:                                list

        """

        # select the appropriate template file
        self.template_file = utx.select_template_file(template_file, extension="xbm")

        # read in data specification yaml
        self.data_specification_file = utx.select_data_specification_file(yaml_file=data_specification_file,
                                                                          extension="xbm")
        self.data_spec_dict = utx.yaml_to_dict(self.data_specification_file)

        # instantiate data specification and validation class
        self.file_spec = modify.Modify(comment_indicator=self.data_spec_dict["comment_indicator"],
                                       data_dict=self.data_spec_dict["data_dict"],
                                       column_widths=self.data_spec_dict["column_widths"],
                                       column_alignment=self.data_spec_dict["column_alignment"],
                                       data_types=self.data_spec_dict["data_types"],
                                       column_list=self.data_spec_dict["column_list"],
                                       value_columns=self.data_spec_dict["value_columns"])

        # prepare template data frame for alteration
        self.template_df, self.template_header = modify.prep_data(field_dict=self.file_spec.data_dict,
                                                                  template_file=self.template_file,
                                                                  column_list=self.file_spec.column_list,
                                                                  column_widths=self.file_spec.column_widths,
                                                                  data_types=self.file_spec.data_types,
                                                                  comment=self.file_spec.comment_indicator,
                                                                  skip_rows=skip_rows,
                                                                  replace_dict={"********": np.nan})


def get_samples(param_dict: dict,
                basin_name: str,
                n_samples: int = 1,
                sampling_method: str = "LHS",
                seed_value: Union[None, int] = None):
    """Generate or load Latin Hypercube Samples (LHS).

    Currently, this reads in an input file of precalculated samples.  This should happen
    on-demand and be given a seed value if reproducibility is needed.

    :param sampling_method:     Sampling method.  Uses SALib's implementation (see https://salib.readthedocs.io/en/latest/).
                                Currently supports the following method:  "LHS" for Latin Hypercube Sampling
    :type sampling_method:      str

    :param n_samples:           Number of LHS samples to generate, optional. Defaults to 1.
    :type n_samples:            int, optional

    :param seed_value:          Seed value to use when generating samples for the purpose of reproducibility.
                                Defaults to None.
    :type seed_value:           Union[None, int], optional

    """

    problem_dict = {'num_vars': 7,
                    'names': ['mu0', 'sigma0', 'mu1', 'sigma1', 'p00', 'p11', 'iwr_multiplier'],
                    'bounds': [[param_dict["mu0"]["lower"], param_dict["mu0"]["upper"]],
                               [param_dict["sigma0"]["lower"], param_dict["sigma0"]["upper"]],
                               [param_dict["mu1"]["lower"], param_dict["mu1"]["upper"]],
                               [param_dict["sigma1"]["lower"], param_dict["sigma1"]["upper"]],
                               [param_dict["p00"]["lower"], param_dict["p00"]["upper"]],
                               [param_dict["p11"]["lower"], param_dict["p11"]["upper"]],
                               [param_dict["iwr_multiplier"][basin_name]["lower"], param_dict["iwr_multiplier"][basin_name]["upper"]]]}

    return sampler.generate_samples(problem_dict=problem_dict,
                                    n_samples=n_samples,
                                    sampling_method=sampling_method,
                                    seed_value=seed_value)


def generate_dry_state_means():
    """Generate or load dry state means.

    Currently, this reads in an input file of precalculated samples.  This should happen
    on-demand and be given a seed value if reproducibility is needed.

    """

    target_file = pkg.resource_filename("statemodify", os.path.join("data", "dry_state_means.txt"))

    return np.loadtxt(target_file)


def generate_wet_state_means(load: bool = True):
    """Generate or load wet state means.

    Currently, this reads in an input file of precalculated samples.  This should happen
    on-demand and be given a seed value if reproducibility is needed.

    """

    target_file = pkg.resource_filename("statemodify", os.path.join("data", "wet_state_means.txt"))

    return np.loadtxt(target_file)


def generate_dry_covariance_matrix(load: bool = True):
    """Generate or load dry covariance matrix.

    Currently, this reads in an input file of precalculated samples.  This should happen
    on-demand and be given a seed value if reproducibility is needed.

    """

    target_file = pkg.resource_filename("statemodify", os.path.join("data", "covariance_matrix_dry.txt"))

    return np.loadtxt(target_file)


def generate_wet_covariance_matrix(load: bool = True):
    """Generate or load wet covariance matrix.

    Currently, this reads in an input file of precalculated samples.  This should happen
    on-demand and be given a seed value if reproducibility is needed.

    """

    target_file = pkg.resource_filename("statemodify", os.path.join("data", "covariance_matrix_wet.txt"))

    return np.loadtxt(target_file)


def generate_transition_matrix(load: bool = True):
    """Generate or load transition matrix.

    Currently, this reads in an input file of precalculated samples.  This should happen
    on-demand and be given a seed value if reproducibility is needed.

    """

    target_file = pkg.resource_filename("statemodify", os.path.join("data", "transition_matrix.txt"))

    return np.loadtxt(target_file)


def calculate_array_monthly(df: pd.DataFrame,
                            value_fields: List[str],
                            year_field: str = "year") -> np.array:
    """Create an array of month rows and site columns for each year in the data frame.

    :param df:                          Input template data frame
    :type df:                           pd.DataFrame

    :param value_fields:                Month fields in data
    :type value_fields:                 List[str]

    :param year_field:                  Name of the year field
    :type year_field:                   str

    :return:                            Array of value for month, site
    :rtype:                             np.array

    """

    out_list = []
    for year in df[year_field].sort_values().unique():

        for month in value_fields:
            v = df.loc[df[year_field] == year][month]

            out_list.append(v.astype(np.int64))

    return np.array(out_list)


def calculate_array_annual(monthly_arr: np.array) -> np.array:
    """Calculate annual values.

    :param monthly_arr:                     Array of monthly xbm values
    :type monthly_arr:                      np.array

    :return:                                Array of year, site
    :rtype:                                 np.array

    """

    n_years = int(monthly_arr.shape[0] / 12)
    n_sites = monthly_arr.shape[1]

    annual_arr = np.zeros([n_years, n_sites])
    for i in range(n_years):
        annual_arr[i, :] = np.sum(monthly_arr[i * 12:(i + 1) * 12], 0)

    return annual_arr


def calculate_annual_sum(arr: np.array, axis: int = 1) -> np.array:
    """Calculate annual sum of input array.

    :param arr:                     Input 2D array where year, site
    :type arr:                      np.array

    :param axis:                    Axis to sum over. Default: 1
    :type axis:                     int

    :return:                        Array of sums
    :rtype:                         np.array

    """

    return np.sum(arr, axis)


def calculate_annual_mean_fractions(arr_annual: np.array,
                                    arr_sum: np.array) -> np.array:
    """Calculate annual mean fractions of total values.

    :param arr_annual:                          Array of annual values per site
    :type arr_annual:                           np.array

    :param arr_sum:                             Array of annual sums
    :type arr_sum:                              np.array

    :return:                                    Array of fractions
    :rtype:                                     np.array

    """

    iwr_fractions = np.zeros(np.shape(arr_annual))

    for i in range(np.shape(arr_annual)[0]):
        iwr_fractions[i, :] = arr_annual[i, :] / arr_sum[i]

    return np.mean(iwr_fractions, 0)


def fit_iwr_model(xbm_data_array_annual: np.array,
                  iwr_data_array_annual: np.array):
    """Model annual irrigation demand anomaly as a function of annual flow anomaly at last node.

    :param xbm_data_array_annual:                       Annual flow from XBM
    :type xbm_data_array_annual:                        np.array

    :param iwr_data_array_annual:                       Annual data from IWR
    :type iwr_data_array_annual:                        np.array

    """

    IWRsums = np.sum(iwr_data_array_annual, 1)
    Qsums = xbm_data_array_annual[:, -1]

    Qsums_prime = Qsums - np.mean(Qsums)
    IWRsums_prime = IWRsums - np.mean(IWRsums)

    # fit model of IWR anomalies as function of Q anomalies
    # (no intercept b/c using anomalies)
    X = np.reshape(Qsums_prime, [len(Qsums_prime), 1])
    y = IWRsums_prime
    model = sm.OLS(y, X).fit()

    # find mean and st dev of residuals, which are normally distributed
    mu = np.mean(model.resid)
    sigma = np.std(model.resid)

    return model.params, mu, sigma


def generate_hmm_inputs(template_file, n_basins=5):
    """Generate HMM input files for all basins."""
    annual_q_h_all = np.array(pd.read_csv(template_file))
    log_annual_q_h = np.log(annual_q_h_all + 1)  # add 1 because some sites have 0 flow

    # fit multi-site HMM to approximately last 2/3 of historical record
    hmm_model = hmm.GMMHMM(n_components=2, n_iter=1000, covariance_type='full').fit(log_annual_q_h[30::, :])

    # Pull out some model parameters
    mus = np.array(hmm_model.means_)
    p = np.array(hmm_model.transmat_)
    hidden_states = hmm_model.predict(log_annual_q_h)

    # Dry state doesn't always come first,but we want it to be, so flip if it isn't
    if mus[0][0][0] > mus[1][0][0]:
        mus = np.flipud(mus)
        p = np.fliplr(np.flipud(p))
        covariance_matrix_dry = hmm_model.covars_[[1]].reshape(n_basins, n_basins)
        covariance_matrix_wet = hmm_model.covars_[[0]].reshape(n_basins, n_basins)
        hidden_states = 1 - hidden_states
    else:
        covariance_matrix_dry = hmm_model.covars_[[0]].reshape(n_basins, n_basins)
        covariance_matrix_wet = hmm_model.covars_[[1]].reshape(n_basins, n_basins)

        # Redefine variables
    dry_state_means = mus[0, :]
    wet_state_means = mus[1, :]
    transition_matrix = p

    return dry_state_means, wet_state_means, covariance_matrix_dry, covariance_matrix_wet, transition_matrix


def generate_flows(dry_state_means: np.array,
                   wet_state_means: np.array,
                   covariance_matrix_dry: np.array,
                   covariance_matrix_wet: np.array,
                   transition_matrix: np.array,
                   mu_0: float,
                   sigma_0: float,
                   mu_1: float,
                   sigma_1: float,
                   p00: float,
                   p11: float,
                   n_basins: int = 5,
                   n_years: int = 105,
                   seed_value: Union[None, int] = None):
    """Generate synthetic streamflow data using a Hidden Markov Model (HMM).

    :param dry_state_means: mean streamflow values for dry state
    :type dry_state_means: np.array

    :param wet_state_means: mean streamflow values for wet state
    :type wet_state_means: np.array

    :param covariance_matrix_dry: covariance matrix for dry state
    :type covariance_matrix_dry: np.array

    :param covariance_matrix_wet: covariance matrix for wet state
    :type covariance_matrix_wet: np.array

    :param transition_matrix: transition matrix for HMM
    :type transition_matrix: np.array

    :param mu_0: mean multiplier for dry state
    :type mu_0: float

    :param sigma_0: covariance multiplier for dry state
    :type sigma_0: float

    :param mu_1: mean multiplier for wet state
    :type mu_1: float

    :param sigma_1: covariance multiplier for wet state
    :type sigma_1: float

    :param p00: transition matrix multiplier for dry state
    :type p00: float

    :param p11: transition matrix multiplier for wet state
    :type p11: float

    :param n_basins: number of sites to generate data for
    :type n_basins: int

    :param n_years: number of years to generate data for
    :type n_years: int

    :param seed_value: random seed value
    :type seed_value: Union[None, int]

    :return: synthetic streamflow data
    :rtype: np.array

    """
    # set random seed if desired
    if seed_value is not None:
        random.seed(seed_value)
        np.random.seed(seed_value)

    # Apply mean multipliers
    dry_state_means_sampled = dry_state_means * mu_0
    wet_state_means_sampled = wet_state_means * mu_1

    # Apply covariance multipliers
    covariance_matrix_dry_sampled = covariance_matrix_dry * sigma_0
    for j in range(n_basins):
        covariance_matrix_dry_sampled[j, j] *= sigma_0

    covariance_matrix_wet_sampled = covariance_matrix_wet * sigma_1
    for j in range(n_basins):
        covariance_matrix_wet_sampled[j, j] *= sigma_1

    # Apply transition matrix multipliers
    transition_matrix_sampled = transition_matrix.copy()
    transition_matrix_sampled[0, 0] += p00
    transition_matrix_sampled[1, 1] += p11
    transition_matrix_sampled[0, 1] = 1 - transition_matrix_sampled[0, 0]
    transition_matrix_sampled[1, 0] = 1 - transition_matrix_sampled[1, 1]

    # calculate stationary distribution to determine unconditional probabilities
    eigenvals, eigenvecs = np.linalg.eig(np.transpose(transition_matrix_sampled))
    one_eigval = np.argmin(np.abs(eigenvals - 1))
    pi = eigenvecs[:, one_eigval] / np.sum(eigenvecs[:, one_eigval])
    unconditional_dry = pi[0]
    unconditional_wet = pi[1]

    log_annual_q_s = np.zeros([n_years, n_basins])

    states = np.empty([np.shape(log_annual_q_s)[0]])
    if random.random() <= unconditional_dry:
        states[0] = 0
        log_annual_q_s[0, :] = np.random.multivariate_normal(np.reshape(dry_state_means_sampled, -1), covariance_matrix_dry_sampled)
    else:
        states[0] = 1
        log_annual_q_s[0, :] = np.random.multivariate_normal(np.reshape(wet_state_means_sampled, -1), covariance_matrix_wet_sampled)

    # generate remaining state trajectory and log space flows
    for j in range(1, np.shape(log_annual_q_s)[0]):
        if random.random() <= transition_matrix_sampled[int(states[j - 1]), int(states[j - 1])]:
            states[j] = states[j - 1]
        else:
            states[j] = 1 - states[j - 1]

        if states[j] == 0:
            log_annual_q_s[j, :] = np.random.multivariate_normal(np.reshape(dry_state_means_sampled, -1), covariance_matrix_dry_sampled)
        else:
            log_annual_q_s[j, :] = np.random.multivariate_normal(np.reshape(wet_state_means_sampled, -1), covariance_matrix_wet_sampled)

    # convert log-space flows to real-space flows
    annual_q_s = np.exp(log_annual_q_s) - 1

    return annual_q_s


def generate_modified_file(source_object: object,
                           monthly_data_array: np.array,
                           output_dir: str,
                           scenario: str,
                           sample_id: int = 0):
    """Generate modified template data frame.

    :param source_object:       Instantiated object containing the source data and file specifics.
    :type source_object:        object

    :param monthly_data_array:  Array of monthly data per year and site matching the shape of the input
                                value columns.
    :type monthly_data_array:   np.array

    :param sample_id:           Numeric ID of sample that is being processed. Defaults to 0.
    :type sample_id:            int

    :param output_dir:          Path to output directory.
    :type output_dir:           str

    :param scenario:            Scenario name.
    :type scenario:             str

    """

    file_spec = source_object.data_spec_dict

    # format for overwriting original data in data frame
    arr_shape = monthly_data_array.shape
    monthly_q_s_frame = monthly_data_array.reshape(arr_shape[0] * arr_shape[1], arr_shape[3])

    # reconstruct precision
    source_object.template_df[file_spec.value_columns] = monthly_q_s_frame.round(0).astype(np.int64)

    # convert all fields to str type
    source_object.template_df = source_object.template_df.astype(str)

    # add in trailing decimal
    source_object.template_df[file_spec.value_columns] = source_object.template_df[file_spec.value_columns] + "."

    # add formatted data to output string
    data = modify.construct_data_string(source_object.template_df,
                                        file_spec.column_list,
                                        file_spec.column_widths,
                                        file_spec.column_alignment)

    # write output file
    output_file = modify.construct_outfile_name(source_object.template_file,
                                                output_dir,
                                                scenario,
                                                sample_id)

    with open(output_file, "w") as out:
        # write header
        out.write(source_object.template_header)

        # write data
        out.write(data)

    return output_file


def modify_single_xbm_iwr(mu_0: float,
                          sigma_0: float,
                          mu_1: float,
                          sigma_1: float,
                          p00: float,
                          p11: float,
                          iwr_multiplier: float,
                          output_dir: str,
                          scenario: str = "",
                          sample_id: int = 0,
                          n_sites: int = 208,
                          n_years: int = 105,
                          n_basins: int = 5,
                          xbm_skip_rows: int = 1,
                          iwr_skip_rows: int = 1,
                          xbm_template_file: Union[None, str] = None,
                          iwr_template_file: Union[None, str] = None,
                          xbm_data_specification_file: Union[None, str] = None,
                          iwr_data_specification_file: Union[None, str] = None,
                          historical_column: int = 0,
                          months_in_year: int = 12,
                          seed_value: Union[None, int] = None):
    """Generate synthetic streamflow data using a Hidden Markov Model (HMM).

    :param mu_0:                            mean multiplier for dry state
    :type mu_0:                             float

    :param sigma_0:                         covariance multiplier for dry state
    :type sigma_0:                          float

    :param mu_1:                            mean multiplier for wet state
    :type mu_1:                             float

    :param sigma_1:                         covariance multiplier for wet state
    :type sigma_1:                          float

    :param p00:                             probability of staying in dry state
    :type p00:                              float

    :param p11:                             probability of staying in wet state
    :type p11:                              float

    :param iwr_multiplier:                  irrigation water requirement multiplier
    :type iwr_multiplier:                   float

    :param sample_id:                       Numeric ID of sample that is being processed. Defaults to 0.
    :type sample_id:                        int

    :param output_dir:                      Path to output directory.
    :type output_dir:                       str

    :param scenario:                        Scenario name.
    :type scenario:                         str

    :param n_sites:                         number of sites
    :type n_sites:                          int

    :param n_years:                         number of years
    :type n_years:                          int

    :param n_basins:                        number of basins in HMM inputs
    :type n_basins:                         int

    :param xbm_skip_rows:                   number of rows to skip in XBM template file
    :type xbm_skip_rows:                    int

    :param iwr_skip_rows:                   number of rows to skip in IWR template file
    :type iwr_skip_rows:                    int

    :param xbm_template_file:               Template file to build XBM adjustment off of
    :type xbm_template_file:                Union[None, str]

    :param iwr_template_file:               Template file to build IWR adjustment off of
    :type iwr_template_file:                Union[None, str]

    :param xbm_data_specification_file:     Specification YAML file for XBM format
    :type xbm_data_specification_file:      Union[None, str]

    :param iwr_data_specification_file:     Specification YAML file for XBM format
    :type iwr_data_specification_file:      Union[None, str]

    :param historical_column:               Index of year to use for historical data
    :type historical_column:                int

    :param months_in_year:                  Number of months in year
    :type months_in_year:                   int

    :param seed_value:                      Integer to use for random seed or None if not desired
    :type seed_value:                       Union[None, int] = None)

    """

    # set random seed if desired
    if seed_value is not None:
        random.seed(seed_value)
        np.random.seed(seed_value)

    # instantiate xbm template data and specification
    xbm = GenerateXbmData(skip_rows=xbm_skip_rows,
                          template_file=xbm_template_file,
                          data_specification_file=xbm_data_specification_file)

    # instantiate iwr template data and specification
    iwr = GenerateIwrData(skip_rows=iwr_skip_rows,
                          template_file=iwr_template_file,
                          data_specification_file=iwr_data_specification_file)

    # calculate the xbm monthly data array
    MonthlyQ_h = calculate_array_monthly(df=xbm.template_df,
                                         value_fields=xbm.data_spec_dict["value_columns"],
                                         year_field="year")

    # generate the xbm yearly data array
    AnnualQ_h = calculate_array_annual(MonthlyQ_h)

    # calculate the iwr monthly data array
    MonthlyIWR_h = calculate_array_monthly(df=iwr.template_df,
                                           value_fields=iwr.data_spec_dict["value_columns"],
                                           year_field="year")

    # generate the iwr yearly data array
    AnnualIWR_h = calculate_array_annual(MonthlyIWR_h)

    # calculate annual sum
    IWRsums_h = calculate_annual_sum(AnnualIWR_h, 1)

    # calculate annual mean fractions
    IWRfractions_h = calculate_annual_mean_fractions(AnnualIWR_h, IWRsums_h)

    # model annual irrigation demand anomaly as function of annual flow anomaly at last node
    BetaIWR, muIWR, sigmaIWR = fit_iwr_model(AnnualQ_h, AnnualIWR_h)

    AnnualQ_s = generate_flows(dry_state_means=generate_dry_state_means(),
                               wet_state_means=generate_wet_state_means(),
                               covariance_matrix_dry=generate_dry_covariance_matrix(),
                               covariance_matrix_wet=generate_wet_covariance_matrix(),
                               transition_matrix=generate_transition_matrix(),
                               mu_0=mu_0,
                               sigma_0=sigma_0,
                               mu_1=mu_1,
                               sigma_1=sigma_1,
                               p00=p00,
                               p11=p11,
                               n_basins=n_basins,
                               n_years=n_years)

    # calculate annual IWR anomalies based on annual flow anomalies at last node
    TotalAnnualIWRanomalies_s = BetaIWR * (AnnualQ_s[:,historical_column]-np.mean(AnnualQ_s[:,historical_column])) + \
            ss.norm.rvs(muIWR, sigmaIWR,len(AnnualQ_s[:,historical_column]))
    TotalAnnualIWR_s = np.mean(IWRsums_h)*iwr_multiplier + TotalAnnualIWRanomalies_s
    AnnualIWR_s = np.outer(TotalAnnualIWR_s, IWRfractions_h)

    # reshape monthly flows at all sites all years
    MonthlyQ_all = MonthlyQ_h.reshape(n_years, months_in_year, n_sites)
    MonthlyQ_all_ratios = np.zeros(np.shape(MonthlyQ_all))

    # Divide monthly flows at each site by the monthly flow at the last node
    MonthlyQ_all_ratios = MonthlyQ_all / MonthlyQ_all[:, :, -1][:, :, np.newaxis]

    # Get historical flow ratios for last node monthly
    last_node_breakdown = MonthlyQ_all[:, :, -1] / AnnualQ_h[:, -1][:, np.newaxis]

    MonthlyQ_s = np.zeros([n_years, n_sites, months_in_year])
    MonthlyIWR_s = np.zeros([n_years, np.shape(MonthlyIWR_h)[1], months_in_year])

    # disaggregate annual flows and demands at all sites using randomly selected neighbor from k nearest based on flow
    dists = np.zeros([n_years, np.shape(AnnualQ_h)[0]])
    for j in range(n_years):
        for m in range(np.shape(AnnualQ_h)[0]):
            dists[j, m] = dists[j, m] + (AnnualQ_s[j, historical_column] - AnnualQ_h[m, -1]) ** 2

    # Create probabilities for assigning a nearest neighbor for the simulated years
    probs = np.zeros([int(np.sqrt(np.shape(AnnualQ_h)[0]))])
    for j in range(len(probs)):
        probs[j] = 1 / (j + 1)
        probs = probs / np.sum(probs)
        for j in range(len(probs) - 1):
            probs[j + 1] = probs[j] + probs[j + 1]
    probs = np.insert(probs, 0, 0)

    for j in range(n_years):

        # select one of k nearest neighbors for each simulated year
        indices = np.argsort(dists[j, :])[0:int(np.sqrt(np.shape(AnnualQ_h)[0]))]
        randnum = random.random()
        for k in range(1, len(probs)):
            if randnum > probs[k - 1] and randnum <= probs[k]:
                neighbor_index = indices[k - 1]

        # Use selected neighbors to downscale flows and demands each year at last nodw
        MonthlyQ_s[j, -1, :] = last_node_breakdown[neighbor_index, :] * AnnualQ_s[j, historical_column]

        # Find monthly flows at all other sites each year
        for k in range(months_in_year):
            MonthlyQ_s[j, :, k] = MonthlyQ_all_ratios[neighbor_index, k, :] * MonthlyQ_s[j, -1, k]

        for k in range(np.shape(MonthlyIWR_h)[1]):
            if np.sum(MonthlyIWR_h[neighbor_index * months_in_year:(neighbor_index + 1) * months_in_year, k]) > 0:
                proportions = MonthlyIWR_h[neighbor_index * months_in_year:(neighbor_index + 1) * months_in_year, k] / np.sum(
                    MonthlyIWR_h[neighbor_index * months_in_year:(neighbor_index + 1) * months_in_year, k])
            else:
                proportions = np.zeros([months_in_year])

            MonthlyIWR_s[j, k, :] = proportions * AnnualIWR_s[j, k]

    # generate modified output files
    xbm_new_file = generate_modified_file(source_object=xbm,
                                          monthly_data_array=MonthlyQ_s,
                                          output_dir=output_dir,
                                          scenario=scenario,
                                          sample_id=sample_id)

    iwr_new_file = generate_modified_file(source_object=iwr,
                                          monthly_data_array=MonthlyIWR_s,
                                          output_dir=output_dir,
                                          scenario=scenario,
                                          sample_id=sample_id)

    return xbm_new_file, iwr_new_file


def modify_xbm_iwr(output_dir: str,
                   scenario: str = "",
                   basin_name: str = "Upper_Colorado",
                   n_years: int = 105,
                   n_basins: int = 5,
                   xbm_skip_rows: int = 1,
                   iwr_skip_rows: int = 1,
                   xbm_template_file: Union[None, str] = None,
                   iwr_template_file: Union[None, str] = None,
                   xbm_data_specification_file: Union[None, str] = None,
                   iwr_data_specification_file: Union[None, str] = None,
                   months_in_year: int = 12,
                   seed_value: Union[None, int] = None,
                   n_jobs: int = -1,
                   n_samples: int = 1):
    """Generate flows for all samples for all basins in parallel to build modified XBM and IWR files.

    :param output_dir:                      Path to output directory.
    :type output_dir:                       str

    :param scenario:                        Scenario name.
    :type scenario:                         str

    :param basin_name:                      Name of basin for either:
                                                Upper_Colorado
                                                Yampa
                                                San_Juan
                                                Gunnison
                                                White
    :type basin_name:                       str

    :param n_years:                         number of years
    :type n_years:                          int

    :param n_basins:                        number of basins in HMM inputs
    :type n_basins:                         int

    :param xbm_skip_rows:                   number of rows to skip in XBM template file
    :type xbm_skip_rows:                    int

    :param iwr_skip_rows:                   number of rows to skip in IWR template file
    :type iwr_skip_rows:                    int

    :param xbm_template_file:               Template file to build XBM adjustment off of
    :type xbm_template_file:                Union[None, str]

    :param iwr_template_file:               Template file to build IWR adjustment off of
    :type iwr_template_file:                Union[None, str]

    :param xbm_data_specification_file:     Specification YAML file for XBM format
    :type xbm_data_specification_file:      Union[None, str]

    :param iwr_data_specification_file:     Specification YAML file for XBM format
    :type iwr_data_specification_file:      Union[None, str]

    :param months_in_year:                  Number of months in year
    :type months_in_year:                   int

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
        basin_name = "Upper_Colorado"

        # seed value for reproducibility if so desired
        seed_value = None

        # number of jobs to launch in parallel; -1 is all but 1 processor used
        n_jobs = -1

        # number of samples to generate
        n_samples = 100

        # generate a batch of files using generated LHS
        stm.modify_xbm_iwr(output_dir=output_directory,
                           scenario=scenario,
                           basin_name=basin_name,
                           seed_value=seed_value,
                           n_jobs=n_jobs,
                           n_samples=n_samples)


    """

    yaml_file = pkg.resource_filename("statemodify", "data/parameter_definitions.yml")
    param_dict = utx.yaml_to_dict(yaml_file)

    # generate an array of samples to process
    sample_array = sampler.generate_sample_all_params(n_samples=n_samples,
                                                      seed_value=seed_value)

    # generate all files in parallel
    results = Parallel(n_jobs=n_jobs, backend="loky")(delayed(modify_single_xbm_iwr)(mu_0=sample[param_dict["mu_0"]["index"]],
                                                                                     sigma_0=sample[param_dict["sigma_0"]["index"]],
                                                                                     mu_1=sample[param_dict["mu_1"]["index"]],
                                                                                     sigma_1=sample[param_dict["sigma_1"]["index"]],
                                                                                     p00=sample[param_dict["p00"]["index"]],
                                                                                     p11=sample[param_dict["p11"]["index"]],
                                                                                     iwr_multiplier=sample[param_dict["iwr_multiplier"][basin_name]["index"]],
                                                                                     n_sites=param_dict["xbm_site_metadata"][basin_name]["n_sites"],
                                                                                     historical_column=param_dict["xbm_site_metadata"][basin_name]["historical_column"],
                                                                                     output_dir=output_dir,
                                                                                     scenario=scenario,
                                                                                     sample_id=sample_id,
                                                                                     n_years=n_years,
                                                                                     n_basins=n_basins,
                                                                                     xbm_skip_rows=xbm_skip_rows,
                                                                                     iwr_skip_rows=iwr_skip_rows,
                                                                                     xbm_template_file=xbm_template_file,
                                                                                     iwr_template_file=iwr_template_file,
                                                                                     xbm_data_specification_file=xbm_data_specification_file,
                                                                                     iwr_data_specification_file=iwr_data_specification_file,
                                                                                     months_in_year=months_in_year,
                                                                                     seed_value=seed_value) for sample_id, sample in enumerate(sample_array))



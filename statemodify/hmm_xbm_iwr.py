import os
import pkg_resources as pkg
from random import random
from typing import List, Union

import numpy as np
import pandas as pd
from scipy import stats as ss
from hmmlearn import hmm
import statsmodels as sm

import statemodify.utils as utx
import statemodify.modify as modify
from statemodify.xbm_iwr import GenerateXbmData, GenerateIwrData


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
                  iwr_data_array_annual: np.array,
                  iwr_annual_sum: np.array):
    """Model annual irrigation demand anomaly as a function of annual flow anomaly at last node.

    :param xbm_data_array_annual:                       Annual flow from XBM
    :type xbm_data_array_annual:                        np.array

    :param iwr_data_array_annual:                       Annual data from IWR
    :type iwr_data_array_annual:                        np.array

    :param iwr_annual_sum:                              Sum of annual IWR array
    :type iwr_annual_sum:                               np.array

    """

    q_sum_array = xbm_data_array_annual[:, -1]

    q_sum_prime = q_sum_array - np.mean(q_sum_array)
    iwr_sum_prime = iwr_annual_sum - np.mean(iwr_annual_sum)

    # fit model of IWR anomalies as function of Q anomalies
    # (no intercept b/c using anomalies)
    X = np.reshape(q_sum_prime, [len(q_sum_prime), 1])
    model = sm.OLS(iwr_sum_prime, X).fit()

    # find mean and st dev of residuals, which are normally distributed
    mu = np.mean(model.resid)
    sigma = np.std(model.resid)

    return model.params, mu, sigma


def hmm_modification(xbm_skip_rows: int = 1,
                     iwr_skip_rows: int = 1,
                     xbm_template_file: Union[None, str] = None,
                     iwr_template_file: Union[None, str] = None,
                     xbm_data_specification_file: Union[None, str] = None,
                     iwr_data_specification_file: Union[None, str] = None):

    # instantiate xbm template data and specification
    xbm = GenerateXbmData(skip_rows=xbm_skip_rows,
                          template_file=xbm_template_file,
                          data_specification_file=xbm_data_specification_file)

    # instantiate iwr template data and specification
    iwr = GenerateIwrData(skip_rows=iwr_skip_rows,
                          template_file=iwr_template_file,
                          data_specification_file=iwr_data_specification_file)

    # calculate the xbm monthly data array
    xbm_data_array_monthly = calculate_array_monthly(df=xbm.template_df,
                                                     value_fields=xbm.data_spec_dict["value_columns"],
                                                     year_field="year")

    # generate the xbm yearly data array
    xbm_data_array_annual = calculate_array_annual(xbm_data_array_monthly)

    # calculate the iwr monthly data array
    iwr_data_array_monthly = calculate_array_monthly(df=iwr.template_df,
                                                     value_fields=iwr.data_spec_dict["value_columns"],
                                                     year_field="year")

    # generate the iwr yearly data array
    iwr_data_array_annual = calculate_array_annual(iwr_data_array_monthly)

    # calculate annual sum
    iwr_annual_sum = calculate_annual_sum(iwr_data_array_annual, 1)

    # calculate annual mean fractions
    iwr_fractions_mean = calculate_annual_mean_fractions(iwr_data_array_annual, iwr_annual_sum)





    # # calculate the maximum xbm flow per year for all users
    # xbm_max_flow_per_year = df.loc[df.groupby(['year'])['total'].idxmax()]["total"].values

    # model annual irrigation demand anomaly as function of annual flow anomaly at last node
    # TODO: revisit in terms of new last node assignment
    beta_iwr, mu_iwr, sigma_iwr = fit_iwr_model(xbm_data_array_annual, iwr_data_array_annual)

    # calculate annual IWR anomalies based on annual flow anomalies at last node
    TotalAnnualIWRanomalies_s = beta_iwr * (AnnualQ_s[:, historical_column] - np.mean(AnnualQ_s[:, historical_column])) + \
                                ss.norm.rvs(mu_iwr, sigma_iwr, len(AnnualQ_s[:, historical_column]))
    TotalAnnualIWR_s = np.mean(IWRsums_h) * LHS[p, iwr_column] + TotalAnnualIWRanomalies_s
    AnnualIWR_s = np.dot(np.reshape(TotalAnnualIWR_s, [np.size(TotalAnnualIWR_s), 1]), \
                         np.reshape(IWRfractions_h, [1, np.size(IWRfractions_h)]))


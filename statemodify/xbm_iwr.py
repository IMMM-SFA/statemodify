import os
import pkg_resources as pkg
from random import random

import numpy as np
import pandas as pd
from scipy import stats as ss
from hmmlearn import hmm


def generate_samples():
    """
    TODO:
    Spaceholder function to generate Latin Hypercube Samples (LHS).

    Currently, this reads in an input file of precalculated samples.  This should happen
    on-demand and be given a seed value if reproducibility is needed.

    """

    target_file = pkg.resource_filename("statemodify", os.path.join("data", "LHsamples_CO.txt"))

    return np.loadtxt(target_file)


def generate_dry_state_means():
    """
    TODO:
    Spaceholder function to generate dry state means.

    Currently, this reads in an input file of precalculated samples.  This should happen
    on-demand and be given a seed value if reproducibility is needed.

    """

    target_file = pkg.resource_filename("statemodify", os.path.join("data", "dry_state_means.txt"))

    return np.loadtxt(target_file)


def generate_wet_state_means():
    """
    TODO:
    Spaceholder function to generate wet state means.

    Currently, this reads in an input file of precalculated samples.  This should happen
    on-demand and be given a seed value if reproducibility is needed.

    """

    target_file = pkg.resource_filename("statemodify", os.path.join("data", "wet_state_means.txt"))

    return np.loadtxt(target_file)


def generate_dry_covariance_matrix():
    """
    TODO:
    Spaceholder function to generate dry covariance matrix.

    Currently, this reads in an input file of precalculated samples.  This should happen
    on-demand and be given a seed value if reproducibility is needed.

    """

    target_file = pkg.resource_filename("statemodify", os.path.join("data", "covariance_matrix_dry.txt"))

    return np.loadtxt(target_file)


def generate_wet_covariance_matrix():
    """
    TODO:
    Spaceholder function to generate wet covariance matrix.

    Currently, this reads in an input file of precalculated samples.  This should happen
    on-demand and be given a seed value if reproducibility is needed.

    """

    target_file = pkg.resource_filename("statemodify", os.path.join("data", "covariance_matrix_wet.txt"))

    return np.loadtxt(target_file)


def generate_transition_matrix():
    """
    TODO:
    Spaceholder function to generate transition matrix.

    Currently, this reads in an input file of precalculated samples.  This should happen
    on-demand and be given a seed value if reproducibility is needed.

    """

    target_file = pkg.resource_filename("statemodify", os.path.join("data", "transition_matrix.txt"))

    return np.loadtxt(target_file)


def generate_flows_parallel():
    """Generate flows for all samples for all basins in parallel.

    """

    # generate an array of samples to process
    samples = generate_samples()

    # generate stationary parameters
    dry_state_means = generate_dry_state_means()
    wet_state_means = generate_wet_state_means()
    covariance_matrix_dry = generate_dry_covariance_matrix()
    covariance_matrix_wet = generate_wet_covariance_matrix()
    transition_matrix = generate_transition_matrix()

    # TODO run generate_flows() in parallel


def generate_flows(sample: np.array,
                   dry_state_means: np.array,
                   wet_state_means: np.array,
                   covariance_matrix_dry: np.array,
                   covariance_matrix_wet: np.array,
                   transition_matrix: np.array,
                   n_parameters: int = 5,
                   n_sites: int = 5,
                   n_years: int = 105) -> np.array:
    """Generate flows for all basins.

    """

    # apply mean multipliers
    dry_state_means_sampled = dry_state_means * sample[0]
    wet_state_means_sampled = wet_state_means * sample[2]

    # apply covariance multipliers
    covariance_matrix_dry_sampled = covariance_matrix_dry * sample[1]
    covariance_matrix_wet_sampled = covariance_matrix_wet * sample[3]

    # apply diagonal multipliers to calculate variance
    for i in range(n_parameters):
        covariance_matrix_dry_sampled[i, i] = covariance_matrix_dry_sampled[i, i] * sample[1]
        covariance_matrix_wet_sampled[i, i] = covariance_matrix_wet_sampled[i, i] * sample[3]

    # apply transition matrix multipliers
    transition_matrix_sampled = transition_matrix.copy()
    transition_matrix_sampled[0, 0] = transition_matrix[0, 0] + sample[4]
    transition_matrix_sampled[1, 1] = transition_matrix[1, 1] + sample[5]
    transition_matrix_sampled[0, 1] = 1 - transition_matrix_sampled[0, 0]
    transition_matrix_sampled[1, 0] = 1 - transition_matrix_sampled[1, 1]

    # calculate stationary distribution to determine unconditional probabilities
    eigenvals, eigenvecs = np.linalg.eig(np.transpose(transition_matrix_sampled))
    one_eigval = np.argmin(np.abs(eigenvals - 1))
    pi = eigenvecs[:, one_eigval] / np.sum(eigenvecs[:, one_eigval])
    unconditional_dry = pi[0]
    unconditional_wet = pi[1]

    log_annual_q_s = np.zeros([n_years, n_sites])
    states = np.empty([np.shape(log_annual_q_s)[0]])

    if random() <= unconditional_dry:
        states[0] = 0
        log_annual_q_s[0, :] = np.random.multivariate_normal(np.reshape(dry_state_means_sampled, -1),
                                                             covariance_matrix_dry_sampled)
    else:
        states[0] = 1
        log_annual_q_s[0, :] = np.random.multivariate_normal(np.reshape(wet_state_means_sampled, -1),
                                                             covariance_matrix_wet_sampled)

    # generate remaining state trajectory and log space flows
    for j in range(1, np.shape(log_annual_q_s)[0]):
        if random() <= transition_matrix_sampled[int(states[j - 1]), int(states[j - 1])]:
            states[j] = states[j - 1]
        else:
            states[j] = 1 - states[j - 1]

        if states[j] == 0:
            log_annual_q_s[j, :] = np.random.multivariate_normal(np.reshape(dry_state_means_sampled, -1),
                                                                 covariance_matrix_dry_sampled)
        else:
            log_annual_q_s[j, :] = np.random.multivariate_normal(np.reshape(wet_state_means_sampled, -1),
                                                                 covariance_matrix_wet_sampled)

    # convert log-space flows to real-space flows
    annual_q_s = np.exp(log_annual_q_s) - 1

    return annual_q_s


def create_xbm_data_frames(filename, firstLine, numSites, abbrev):

    # split data on periods
    with open(filename, 'r') as f:
        all_split_data = [x.split('.') for x in f.readlines()]

    f.close()

    numYears = int((len(all_split_data) - firstLine) / numSites)
    MonthlyQ = np.zeros([12 * numYears, numSites])
    for i in range(numYears):
        for j in range(numSites):
            index = firstLine + i * numSites + j
            all_split_data[index][0] = all_split_data[index][0].split()[2]
            MonthlyQ[i * 12:(i + 1) * 12, j] = np.asfarray(all_split_data[index][0:12], float)

    np.savetxt('./Data/' + abbrev + '2015_StateMod/MonthlyQ.csv', MonthlyQ, fmt='%d', delimiter=',')

    # calculate annual flows
    AnnualQ = np.zeros([numYears, numSites])
    for i in range(numYears):
        AnnualQ[i, :] = np.sum(MonthlyQ[i * 12:(i + 1) * 12], 0)

    np.savetxt('./Data/' + abbrev + '2015_Statemod/AnnualQ.csv', AnnualQ, fmt='%d', delimiter=',')

    return None


def create_iwr_data_frames(filename, firstLine, numSites, abbrev):
    # split data on periods
    with open(filename, 'r') as f:
        all_split_data = [x.split('.') for x in f.readlines()]

    f.close()

    numYears = int((len(all_split_data) - firstLine) / numSites)
    MonthlyIWR = np.zeros([12 * numYears, numSites])
    for i in range(numYears):
        for j in range(numSites):
            index = firstLine + i * numSites + j
            all_split_data[index][0] = all_split_data[index][0].split()[2]
            MonthlyIWR[i * 12:(i + 1) * 12, j] = np.asfarray(all_split_data[index][0:12], float)

    np.savetxt('./Data/' + abbrev + '2015_StateMod/MonthlyIWR.csv', MonthlyIWR, fmt='%d', delimiter=',')

    # calculate annual flows
    AnnualIWR = np.zeros([numYears, numSites])
    for i in range(numYears):
        AnnualIWR[i, :] = np.sum(MonthlyIWR[i * 12:(i + 1) * 12], 0)

    np.savetxt('./Data/' + abbrev + '2015_StateMod/AnnualIWR.csv', AnnualIWR, fmt='%d', delimiter=',')

    return None


def fit_iwr_model(AnnualQ, AnnualIWR):
    IWRsums = np.sum(AnnualIWR, 1)
    Qsums = AnnualQ[:, -1]

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


def organize_monthly(filename, firstLine, numSites):
    # read in all monthly flows and re-organize into nyears x 12 x nsites matrix
    with open(filename, 'r') as f:
        all_split_data = [x.split('.') for x in f.readlines()]

    f.close()

    numYears = int((len(all_split_data) - firstLine) / numSites)
    MonthlyQ = np.zeros([12 * numYears, numSites])
    sites = []
    for i in range(numYears):
        for j in range(numSites):
            index = firstLine + i * numSites + j
            sites.append(all_split_data[index][0].split()[1])
            all_split_data[index][0] = all_split_data[index][0].split()[2]
            MonthlyQ[i * 12:(i + 1) * 12, j] = np.asfarray(all_split_data[index][0:12], float)

    MonthlyQ = np.reshape(MonthlyQ, [int(np.shape(MonthlyQ)[0] / 12), 12, numSites])

    return MonthlyQ


def write_files(filename, abbrev, firstLine, sampleNo, realizationNo, allMonthlyData, out_directory):
    nSites = np.shape(allMonthlyData)[1]

    # split data on periods
    with open('./Data/' + abbrev + '2015_StateMod/StateMod/' + filename, 'r') as f:
        all_split_data = [x.split('.') for x in f.readlines()]

    f.close()

    # get unsplit data to rewrite firstLine # of rows
    with open('./Data/' + abbrev + '2015_StateMod/StateMod/' + filename, 'r') as f:
        all_data = [x for x in f.readlines()]

    f.close()

    # replace former flows with new flows
    new_data = []
    for i in range(len(all_split_data) - firstLine):
        year_idx = int(np.floor(i / (nSites)))
        # print(year_idx)
        site_idx = np.mod(i, (nSites))
        # print(site_idx)
        row_data = []
        # split first 3 columns of row on space and find 1st month's flow
        row_data.extend(all_split_data[i + firstLine][0].split())
        row_data[2] = str(int(allMonthlyData[year_idx, site_idx, 0]))
        # find remaining months' flows
        for j in range(11):
            row_data.append(str(int(allMonthlyData[year_idx, site_idx, j + 1])))

        # find total flow
        row_data.append(str(int(np.sum(allMonthlyData[year_idx, site_idx, :]))))

        # append row of adjusted data
        new_data.append(row_data)

    f = open(out_directory + filename[0:-4] + '_S' + str(sampleNo) + '_R' + str(realizationNo) + filename[-4::], 'w')
    # write firstLine # of rows as in initial file
    for i in range(firstLine):
        f.write(all_data[i])

    for i in range(len(new_data)):
        # write year, ID and first month of adjusted data
        f.write(new_data[i][0] + ' ' + new_data[i][1] + (19 - len(new_data[i][1]) - len(new_data[i][2])) * ' ' +
                new_data[i][2] + '.')
        # write all but last month of adjusted data
        for j in range(len(new_data[i]) - 4):
            f.write((7 - len(new_data[i][j + 3])) * ' ' + new_data[i][j + 3] + '.')

        # write last month of adjusted data
        if filename[-4::] == '.xbm':
            if len(new_data[i][-1]) <= 7:
                f.write((7 - len(new_data[i][-1])) * ' ' + new_data[i][-1] + '.' + '\n')
            else:
                f.write('********\n')
        else:
            f.write((9 - len(new_data[i][-1])) * ' ' + new_data[i][-1] + '.' + '\n')

    f.close()

    return None


def flow_disaggregation(basin_name: str,
                        basin_abbrev: str,
                        n_parameters: int,
                        n_years: int,
                        n_sites: int,
                        n_iwr_sites: int,
                        start_xbm: int,
                        start_iwr: int,
                        xbm_file: str,
                        iwr_file: str,
                        xbm_out: str,
                        iwr_out: str,
                        site_abbrev: str,
                        abbrev_file_xbm: str,
                        abbrev_file_iwr: str,
                        historical_column: int = 0):
    """Monthly and spatial disaggregation and irrigation files.

    """

    LHS = generate_samples()
    dry_state_means = generate_dry_state_means()
    wet_state_means = generate_wet_state_means()
    covariance_matrix_dry = generate_dry_covariance_matrix()
    covariance_matrix_wet = generate_wet_covariance_matrix()
    transition_matrix = generate_transition_matrix()

    AnnualQ_s = generate_flows(sample=LHS,
                                dry_state_means=dry_state_means,
                                wet_state_means=wet_state_means,
                                covariance_matrix_dry=covariance_matrix_dry,
                                covariance_matrix_wet=covariance_matrix_wet,
                                transition_matrix=transition_matrix,
                                n_parameters=n_parameters,
                                n_sites=n_sites,
                                n_years=n_years)

    # create annual and monthly streamflow dataframes from xbm file
    create_xbm_data_frames(xbm_file, start_xbm, n_sites, site_abbrev)

    # load annual and monthly flow files

    # TODO: why is monthly never used?  Does it need to be in here?
    MonthlyQ_h = np.array(pd.read_csv('./Data/' + site_abbrev + '2015_StateMod/MonthlyQ.csv', header=None))
    AnnualQ_h = np.array(pd.read_csv('./Data/' + site_abbrev + '2015_Statemod/AnnualQ.csv', header=None))

    # Create annual and monthly irrigation dataframes from IWR files
    create_iwr_data_frames(iwr_file, start_iwr, n_iwr_sites, site_abbrev)

    # load historical (_h) irrigation demand data
    AnnualIWR_h = np.loadtxt('./Data/' + site_abbrev + '2015_StateMod/AnnualIWR.csv', delimiter=',')
    MonthlyIWR_h = np.loadtxt('./Data/' + site_abbrev + '2015_StateMod/MonthlyIWR.csv', delimiter=',')
    IWRsums_h = np.sum(AnnualIWR_h, 1)
    IWRfractions_h = np.zeros(np.shape(AnnualIWR_h))
    for i in range(np.shape(AnnualIWR_h)[0]):
        IWRfractions_h[i, :] = AnnualIWR_h[i, :] / IWRsums_h[i]

    IWRfractions_h = np.mean(IWRfractions_h, 0)

    # model annual irrigation demand anomaly as function of annual flow anomaly at last node
    BetaIWR, muIWR, sigmaIWR = fit_iwr_model(AnnualQ_h, AnnualIWR_h)

    # calculate annual IWR anomalies based on annual flow anomalies at last node
    TotalAnnualIWRanomalies_s = BetaIWR * (AnnualQ_s[:, historical_column] - np.mean(AnnualQ_s[:, historical_column])) + \
                                ss.norm.rvs(muIWR, sigmaIWR, len(AnnualQ_s[:, historical_column]))
    TotalAnnualIWR_s = np.mean(IWRsums_h) * LHS[p, 6] + TotalAnnualIWRanomalies_s
    AnnualIWR_s = np.dot(np.reshape(TotalAnnualIWR_s, [np.size(TotalAnnualIWR_s), 1]), np.reshape(IWRfractions_h, [1, np.size(IWRfractions_h)]))

    # Read in monthly flows at all sites
    MonthlyQ_all = organize_monthly(xbm_file, start_xbm, n_sites)
    MonthlyQ_all_ratios = np.zeros(np.shape(MonthlyQ_all))

    # Divide monthly flows at each site by the monthly flow at the last node
    for i in range(np.shape(MonthlyQ_all_ratios)[2]):
        MonthlyQ_all_ratios[:, :, i] = MonthlyQ_all[:, :, i] / MonthlyQ_all[:, :, -1]

    # Get historical flow ratios
    AnnualQ_h_ratios = np.zeros(np.shape(AnnualQ_h))
    for i in range(np.shape(AnnualQ_h_ratios)[0]):
        AnnualQ_h_ratios[i, :] = AnnualQ_h[i, :] / np.sum(AnnualQ_h[i, -1])

    # Get historical flow ratios for last node monthly
    last_node_breakdown = np.zeros([105, 12])
    for i in range(np.shape(last_node_breakdown)[0]):
        last_node_breakdown[i, :] = MonthlyQ_all[i, :, -1] / AnnualQ_h[i, -1]

    MonthlyQ_s = np.zeros([n_years, n_sites, 12])
    MonthlyIWR_s = np.zeros([n_years, np.shape(MonthlyIWR_h)[1], 12])
    # disaggregate annual flows and demands at all sites using randomly selected neighbor from k nearest based on flow
    dists = np.zeros([n_years, np.shape(AnnualQ_h)[0]])
    for j in range(n_years):
        for m in range(np.shape(AnnualQ_h)[0]):
            dists[j, m] = dists[j, m] + (AnnualQ_s[j, 0] - AnnualQ_h[m, -1]) ** 2

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
        neighbors = np.sort(dists[j, :])[0:int(np.sqrt(np.shape(AnnualQ_h)[0]))]
        indices = np.argsort(dists[j, :])[0:int(np.sqrt(np.shape(AnnualQ_h)[0]))]
        randnum = random()
        for k in range(1, len(probs)):
            if randnum > probs[k - 1] and randnum <= probs[k]:
                neighbor_index = indices[k - 1]
        # Use selected neighbors to downscale flows and demands each year at last nodw
        MonthlyQ_s[j, -1, :] = last_node_breakdown[neighbor_index, :] * AnnualQ_s[j, 0]

        # Find monthly flows at all other sites each year
        for k in range(12):
            MonthlyQ_s[j, :, k] = MonthlyQ_all_ratios[neighbor_index, k, :] * MonthlyQ_s[j, -1, k]

        for k in range(np.shape(MonthlyIWR_h)[1]):
            if np.sum(MonthlyIWR_h[neighbor_index * 12:(neighbor_index + 1) * 12, k]) > 0:
                proportions = MonthlyIWR_h[neighbor_index * 12:(neighbor_index + 1) * 12, k] / np.sum(
                    MonthlyIWR_h[neighbor_index * 12:(neighbor_index + 1) * 12, k])
            else:
                proportions = np.zeros([12])

            MonthlyIWR_s[j, k, :] = proportions * AnnualIWR_s[j, k]

    # write new flows to file for LHsample i (inputs: filename, firstLine, sampleNo,realization, allMonthlyFlows,output folder)
    write_files(abbrev_file_xbm, abbrev_file_xbm, start_xbm, i, 1, MonthlyQ_s, xbm_out)

    # write new irrigation demands to file for LHsample i
    write_files(abbrev_file_iwr, abbrev_file_iwr, start_iwr, i, 1, MonthlyIWR_s, iwr_out)








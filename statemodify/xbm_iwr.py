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

    # apply diagonal multipliers
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


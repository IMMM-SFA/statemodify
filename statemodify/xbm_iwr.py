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



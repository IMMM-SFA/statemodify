import os
import pkg_resources
from typing import Union

from hmmlearn import hmm
import pandas as pd
from random import random
import numpy as np
import matplotlib
import matplotlib.cm as cm
from matplotlib.cm import ScalarMappable
from matplotlib import pyplot as plt
import seaborn as sns


def hmm_multisite_fit(n_basins: int = 5,
                      save_parameters: bool = False,
                      output_directory: Union[None, str] = None):

    all_basins_file = pkg_resources.resource_filename("statemodify", "data/all_basins.csv")
    AnnualQ_h_all = pd.read_csv(all_basins_file).values

    # add 1 because some sites have 0 flow
    logAnnualQ_h = np.log(AnnualQ_h_all+1)

    # fit multi-site HMM to approximately last 2/3 of historical record (30 years)
    hmm_model = hmm.GMMHMM(n_components=2,
                           n_iter=1000,
                           covariance_type='full').fit(logAnnualQ_h[30::, :])

    # Pull out some model parameters
    mus = np.array(hmm_model.means_)
    P = np.array(hmm_model.transmat_)
    hidden_states = hmm_model.predict(logAnnualQ_h)

    # Dry state doesn't always come first,but we want it to be, so flip if it isn't
    if mus[0][0][0] > mus[1][0][0]:
        mus = np.flipud(mus)
        P = np.fliplr(np.flipud(P))
        covariance_matrix_dry = hmm_model.covars_[[1]].reshape(n_basins, n_basins)
        covariance_matrix_wet = hmm_model.covars_[[0]].reshape(n_basins, n_basins)
        hidden_states = 1 - hidden_states
    else:
        covariance_matrix_dry = hmm_model.covars_[[0]].reshape(n_basins, n_basins)
        covariance_matrix_wet = hmm_model.covars_[[1]].reshape(n_basins, n_basins)

    # Redefine variables
    dry_state_means = mus[0, :]
    wet_state_means = mus[1, :]
    transition_matrix = P

    if save_parameters:

        if output_directory is None:
            raise ValueError("If saving parameters you must pass a valid path to `output_directory'")

        np.savetxt(os.path.join(output_directory, 'dry_state_means.txt', dry_state_means))
        np.savetxt(os.path.join(output_directory, 'wet_state_means.txt', wet_state_means))
        np.savetxt(os.path.join(output_directory, 'covariance_matrix_dry.txt', covariance_matrix_dry))
        np.savetxt(os.path.join(output_directory, 'covariance_matrix_wet.txt', covariance_matrix_wet))
        np.savetxt(os.path.join(output_directory, 'transition_matrix.txt', transition_matrix))

    # calculate stationary distribution to determine unconditional probabilities
    eigenvals, eigenvecs = np.linalg.eig(np.transpose(transition_matrix))
    one_eigval = np.argmin(np.abs(eigenvals-1))
    pi = eigenvecs[:, one_eigval] / np.sum(eigenvecs[:, one_eigval])
    unconditional_dry = pi[0]
    unconditional_wet = pi[1]

    return {
        "unconditional_dry": unconditional_dry,
        "unconditional_wet": unconditional_wet,
        "logAnnualQ_h": logAnnualQ_h,
        "transition_matrix": transition_matrix,
        "covariance_matrix_wet": covariance_matrix_wet,
        "covariance_matrix_dry": covariance_matrix_dry,
        "wet_state_means": wet_state_means,
        "dry_state_means": dry_state_means
    }


def hmm_multisite_sample (logAnnualQ_h,
                         transition_matrix,
                         unconditional_dry,
                         dry_state_means,
                         wet_state_means,
                         covariance_matrix_dry,
                         covariance_matrix_wet,
                         n_basins: int = 5,
                         n_alternatives: int = 100,
                         save_samples: bool = True,
                         output_directory: Union[None, str] = None):

    for i in range(n_alternatives):

        # Determine the number of years to simulate. Here we are just simulating 105 years across all basins
        logAnnualQ_s = np.zeros([np.shape(logAnnualQ_h)[0], n_basins])

        states = np.empty([np.shape(logAnnualQ_h)[0]])

        if random() <= unconditional_dry:
            states[0] = 0
            logAnnualQ_s[0, :] = np.random.multivariate_normal(np.reshape(dry_state_means, -1), covariance_matrix_dry)
        else:
            states[0] = 1
            logAnnualQ_s[0, :] = np.random.multivariate_normal(np.reshape(wet_state_means, -1), covariance_matrix_wet)

        # generate remaining state trajectory and log space flows
        for j in range(1, np.shape(logAnnualQ_h)[0]):
            if random() <= transition_matrix[int(states[j - 1]), int(states[j - 1])]:
                states[j] = states[j - 1]
            else:
                states[j] = 1 - states[j - 1]

            if states[j] == 0:
                logAnnualQ_s[j, :] = np.random.multivariate_normal(np.reshape(dry_state_means, -1),
                                                                   covariance_matrix_dry)
            else:
                logAnnualQ_s[j, :] = np.random.multivariate_normal(np.reshape(wet_state_means, -1),
                                                                   covariance_matrix_wet)

        AnnualQ_s = np.exp(logAnnualQ_s)

        if save_samples:

            if output_directory is None:
                raise ValueError("If saving samples you must pass a valid path to `output_directory'")

            np.savetxt(os.path.join(output_directory, f"AnnualQ_s{i}.txt"), AnnualQ_s)


def plot_flow_duration_curves(flow_realizations_directory: str,
                              save_figure: bool = False,
                              output_directory: Union[None, str] = None,
                              figure_name: Union[None, str] = None,
                              dpi: int = 300):
    """Plots flow duration curves for historical and synthetic data

    :param flow_realizations_directory:     Full path to the directory containing the flow realization files for each
                                            sample.  E.g., AnnualQ_s0.txt files produced by the 'hmm_multisite_sample'
                                            function.
    :type flow_realizations_directory:      str

    """

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 6))

    # Plot historical FDCs
    all_basins_file = pkg_resources.resource_filename("statemodify", "data/all_basins.csv")
    historical_data = np.loadtxt(all_basins_file, delimiter=',', skiprows=1)
    basins = ['Upper Colorado', 'Gunnison', 'Yampa', 'White', 'San Juan / Dolores']

    for i in range(len(basins)):
        axes.flatten()[i].plot(np.arange(len(historical_data[:, 0])), np.sort(np.log(historical_data[:, i])),
                               color='darkblue')
        axes.flatten()[i].set_xlim([0, 105])
        axes.flatten()[i].set_xlabel('Non-Exceedance %')
        axes.flatten()[i].set_ylabel('log(Streamflow) (log(acft))')
        axes.flatten()[i].set_title(basins[i])
    axes.flatten()[5].axis('off')

    # read synthetic record
    for r in range(100):
        target_sample_file = os.path.join(flow_realizations_directory, f"AnnualQ_s{r}.txt")
        AnnualQ_s = np.loadtxt(target_sample_file)

        for i in range(5):
            axes.flatten()[i].plot(np.arange(len(AnnualQ_s[:,i])),
                                   np.sort(np.log(AnnualQ_s[:,i])),
                                   color = 'cornflowerblue',
                                   alpha=.05)

    for i in range(5):
        axes.flatten()[i].legend(['Historical', 'HMM Traces'])

    plt.tight_layout()

    if save_figure:

        if output_directory is None:
            raise ValueError(f"If choosing to save figure you must specify an 'output_directory'")

        if figure_name is None:
            raise ValueError(f"If choosing to save figure you must specify a 'figure_name'")

        figure_outfile = f"{os.path.join(output_directory, figure_name)}.png"

        plt.savefig(figure_outfile, dpi=dpi)

    plt.show()
    plt.close()


def read_xre(xre_path: str, reservoir_name: str):
    """Reads xre files generated by statemod using HMM data and historical data

    :param xre_path:            str, path to xre files
    :param reservoir_name:      str, name of reservoir that begins XRE file

    :returns:                   a numpy array with reservoir storage across all hmm realizations,
                                a numpy array with reservoir storage from the historical record
                                a numpy array with monthly means of storage from the hist record
                                a numpy array with monthly 1st percentiles of storage of the hist record
    """
    target_columns = [
        'Res ID', 'ACC', 'Year', 'MO', 'Init. Storage',
        'From River By Priority', 'From River By Storage',
        'From River By Other', 'From River By Loss', 'From Carrier By Priority',
        'From Carrier By Other', 'From Carrier By Loss', 'Total Supply',
        'From Storage to River For Use', 'From Storage to River for Exc',
        'From Storage to Carrier for use', 'Total Release', 'Evap',
        'Seep and Spill', 'EOM Content', 'Target Stor', 'Decree Lim',
        'River Inflow', 'River Release', 'River Divert', 'River by Well',
        'River Outflow'
    ]

    for r in range(100):
        input_file = os.path.join(xre_path, f"{reservoir_name}_xre_data_S{r}_1.csv")
        reservoir_data = pd.read_csv(input_file, index_col=False, usecols=target_columns)

        account_0 = reservoir_data[reservoir_data['ACC']==0]
        # TODO:  there is no field named `TOT` in the data but this will return
        monthly_data = account_0[account_0['MO'] != 'TOT']
        monthly_storage = np.array(monthly_data['Init. Storage'].astype(float))
        monthly_storage = np.reshape(monthly_storage, [105, 12])

        if r == 0:
            all_realizations_storage = monthly_storage
        else:
            all_realizations_storage = np.vstack((all_realizations_storage, monthly_storage))

    # read in "historical"
    reservoir_data_hist = pd.read_csv(os.path.join(xre_path, f"{reservoir_name}_xre_data_hist.csv"),
                                      index_col=False)

    account_0_hist = reservoir_data_hist[reservoir_data_hist['ACC']==0]
    monthly_data_hist = account_0_hist[account_0_hist['MO'] != 'TOT']
    monthly_data_hist = np.array(monthly_data_hist['Init. Storage'].astype(float))
    monthly_storage_hist = np.reshape(monthly_data_hist, [105, 12])

    # get mean and first percentile
    hist_means = np.zeros([12])
    hist_1p = np.zeros([12])
    for m in range(12):
        hist_means[m] = np.mean(monthly_storage_hist[:,m])
        hist_1p[m] = np.percentile(monthly_storage_hist[:,m], 0.01)

    return all_realizations_storage, monthly_storage_hist, hist_means, hist_1p


def plot_res_quantiles(hmm_data: np.array,
                       historical_mean: np.array,
                       reservoir_name: str,
                       save_figure: bool = False,
                       output_directory: Union[None, str] = None,
                       dpi: int = 300):
    """Plots quantiles of the HMM data and the mean and 1st percentile of the historical record

    :param hmm_data:            a numpy array with monthly storage data from HMM realizations
    :param historical_mean:     a numpy array with the historical mean monthly storage
    :param res_name:            a string with the name of the reservoir

    """

    months = ['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep']
    ps = np.arange(0, 1.01, 0.05) * 100

    fig, ax = plt.subplots()
    for j in range(1, len(ps)):
        u = np.percentile(hmm_data, ps[j], axis=0)
        l = np.percentile(hmm_data, ps[j - 1], axis=0)
        ax.fill_between(np.arange(12),
                        l,
                        u,
                        color=cm.BrBG(ps[j - 1] / 100.0),
                        alpha=0.75,
                        edgecolor='none')

    ax.plot(np.arange(12), historical_mean, linestyle='--', color='k', linewidth=3)
    ax.set_xlim(0,11)
    ax.set_xticks(np.arange(12))
    ax.set_xticklabels(months)
    ax.set_title(reservoir_name)
    ax.set_ylabel('Storage (acft)')

    my_cmap = plt.cm.get_cmap('BrBG')
    sm = ScalarMappable(cmap=my_cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm)
    cbar.set_label('Percentile of HMM Records', rotation=270, labelpad=20, fontsize=14)
    plt.tight_layout()

    if save_figure:

        if output_directory is None:
            raise ValueError(f"If choosing to save figure you must specify an 'output_directory'")

        figure_outfile = os.path.join(output_directory, f"{reservoir_name}_HMM_Hist_quantiles.png")

        plt.savefig(figure_outfile, dpi=dpi)

    plt.show()
    plt.close()


def plot_reservoir_boxes(hmm_data: np.array,
                         historical_data: np.array,
                         reservoir_name: str,
                         save_figure: bool = False,
                         output_directory: Union[None, str] = None,
                         dpi: int = 300):
    """

    makes a boxplot comparing the historical record to the HMM realizations

    :param hmm_data:                a numpy array with monthly storage data from HMM realizations
    :param historical_data:         a numpy array with monthly storage data from hist data
    :param reservoir_name:          a string with the name of the reservoir

    """

    months = ['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep']

    fig, ax = plt.subplots()

    # HMM
    hmm_box = ax.boxplot(hmm_data, patch_artist=True, positions=np.arange(0, 24, 2), showfliers=False)

    # set up colors
    plt.setp(hmm_box["boxes"], facecolor='lightblue', edgecolor='white')
    plt.setp(hmm_box["whiskers"], color='lightblue')
    plt.setp(hmm_box["caps"], color='lightblue')
    plt.setp(hmm_box["medians"], color='darkblue')

    # Historical
    hist_box = ax.boxplot(historical_data, patch_artist=True, positions=np.arange(1, 25, 2))
    # set up colors
    plt.setp(hist_box["boxes"], facecolor='orange', edgecolor='white')
    plt.setp(hist_box["whiskers"], color='orange')
    plt.setp(hist_box["caps"], color='orange')
    plt.setp(hist_box["fliers"], color='orange')
    plt.setp(hist_box["medians"], color='firebrick')

    # set up labeling
    ax.set_ylabel('Storage (acft)')
    ax.set_xticks(np.arange(0.5, 24, 2))
    ax.set_xticklabels(months)
    ax.set_title(reservoir_name)
    ax.set_ylim([0, max(hmm_data[:, 0]) * 1.1])
    plt.tight_layout()

    if save_figure:

        if output_directory is None:
            raise ValueError(f"If choosing to save figure you must specify an 'output_directory'")

        figure_outfile = os.path.join(output_directory, f"{reservoir_name}_HMM_Hist_boxplot.png")

        plt.savefig(figure_outfile, dpi=dpi)

    plt.show()
    plt.close()

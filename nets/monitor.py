#!/usr/bin/env python
import sys

import click
import pandas as pd
import numpy as np

from scipy.special import betainc


import sys

import click
import pandas as pd
import numpy as np

from scipy.special import betainc

OUTCOME_COL_NAMES = ['seed_idx', 'basin_id']
# the key given in the outcomes table corresponding to either
# source or sink basin
SOURCE_KEY = 1
SINK_KEY = 2

def read_outcome(outcome_path):
    """Read in an outcomes file depending on the format (either space or
    comma delimited) and return a dataframe."""

    good_format = False
    with open(outcome_path, 'r') as rf:
        if rf.readline().strip() == ','.join(OUTCOME_COL_NAMES):
            good_format = True

    if good_format:
        print("Using csv format")
        outcome_df = pd.read_csv(outcome_path,
                                 index_col=0)
    else:
        print("Using space format")
        outcome_df = pd.read_csv(outcome_path,
                                 names=OUTCOME_COL_NAMES,
                                 header=None,
                                 delim_whitespace=True)

    return outcome_df

def aggregate_outcomes(outcome_paths):
    """Read in and aggregate multiple outcomes files into one dataframe. """

    outcome_dfs = []
    for outcome_path in outcome_paths:
        outcome_df = read_outcome(outcome_path)
        outcome_dfs.append(outcome_df)

    # concatenate the tables into a master outcomes file with an index
    # for each individual outcome
    outcomes_df = pd.concat(outcome_dfs).reset_index(drop=True)

    return outcomes_df


def init_monitor_df(seeds_df):
    """Given a seeds dataframe (seed_idx, committor_prob, bin_idx, ...)
    initialize a dataframe aggregating by bin_idx for reporting bin
    statistics.

    """

    # initialize the monitor dataframe
    bin_idxs = []
    bin_l_edges = []
    bin_centers = []
    bin_u_edges = []
    for bin_idx, bin_df in seeds_df.groupby('bin_idx'):
        bin_idxs.append(bin_idx)
        bin_l_edges.append(bin_df['bin_l_edge'].values[0])
        bin_centers.append(bin_df['bin_center'].values[0])
        bin_u_edges.append(bin_df['bin_u_edge'].values[0])

    monitor_df = pd.DataFrame({'bin_idx' : bin_idxs,
                               'bin_l_edge' : bin_l_edges,
                               'bin_center' : bin_centers,
                               'bin_u_edge' : bin_u_edges})
    return monitor_df

def count_outcomes(outcomes_df, n_seeds):
    """For an outcomes dataframe and the number of seeds count the
    outcomes for each basin.

    Returns:
    num_source : int
    num_sink : int

    """

    # count up the outcomes for each group
    num_source = np.zeros(n_seeds)
    num_sink = np.zeros(n_seeds)

    # count the number of completed seeds for their results
    for nets_idx, row in outcomes_df.iterrows():
        if row['basin_id'] == SOURCE_KEY:
            num_source[row['seed_idx']] += 1
        elif row['basin_id'] == SINK_KEY:
            num_sink[row['seed_idx']] += 1

    return num_source, num_sink

def bin_seeds(monitor_df, seeds_df):
    bin_n_sources = []
    bin_n_sinks = []
    for bin_idx, row in monitor_df.iterrows():
        l_edge = row['bin_l_edge']
        u_edge = row['bin_u_edge']

        seeds = seeds_df[(l_edge < seeds_df['committor_prob']) &
                             (seeds_df['committor_prob'] <= u_edge)]
        bin_n_source = seeds['n_source'].sum()
        bin_n_sink = seeds['n_sink'].sum()

        bin_n_sources.append(bin_n_source)
        bin_n_sinks.append(bin_n_sink)

    return bin_n_source, bin_n_sink

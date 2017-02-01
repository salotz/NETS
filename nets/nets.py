import sys

import click

import pandas as pd
import numpy as np
from scipy.special import betainc

import nets.monitor
import nets.seed

@click.group()
def cli():
    pass

@cli.command('seed')
@click.option('--assignments', type=click.Path(exists=True),
                help="Data-structure mapping trajectory->[cluster_id, ...] for each frame"
                "of the trajectory. This is the default dataset of MSMBuilder clustering.")
@click.argument('clusters', nargs=1, type=click.Path(exists=True))
def seed(assignments):
    pass

@cli.command('monitor')
@click.option('-O', 'output', default=None, type=click.Path(),
              help="file to write out the info table if given")
@click.option('--agg-outcomes', 'outcomes_target', default=None, type=click.Path(),
              help="file to write out the aggregate outcomes if given")
@click.option('--beta-upper', 'beta_upper_x', default=0.4, type=float,
              help="the upper bound of where to integrate the scipy.special.betainc"
                   "function for calculating probability of a true committor"
                   "within the range given the outcomes. Defaults to around"
                   "the transition state.")
@click.option('--beta-lower', 'beta_lower_x', default=0.6, type=float,
              help="the lower bound... ditto")
@click.option('--seed-committors', 'seed_committors', type=click.Path(),
              help="A csv with the seed idx, forward committor for each seed, e.g. 0, 0.43")
@click.argument('outcomes', nargs=-1, type=click.Path(exists=True))
def monitor(output, outcomes_target, beta_upper_x, beta_lower_x,
            seed_committors, outcomes):

    assert seed_committors, "seed-committors must be given"

    # read the seed committors
    seeds_df = pd.read_csv(seed_committors, index_col=0)

    # initialize the monitor dataframe
    monitor_df = nets.monitor.init_monitor_df(seeds_df)

    # aggregate all the outcomes files
    outcomes_df = nets.monitor.aggregate_outcomes(outcomes)

    # make sure the table is not empty, if it is quit
    if len(outcomes_df.index) == 0:
        raise ValueError("No values in the outcomes file")

    num_source, num_sink = nets.monitor.count_outcomes(outcomes_df, seeds_df.shape[0])

    seeds_df['n_source'] = num_source
    seeds_df['n_sink'] = num_sink

    # bin the seeds by their starting committor values
    bin_n_source_col, bin_n_sink_col = nets.monitor.bin_seeds(monitor_df, seeds_df)

    # add this information to the monitor_df
    monitor_df['n_source'] = bin_n_source_col
    monitor_df['n_sink'] = bin_n_sink_col

    ## Calculate other things for the monitor_df
    # a column for the raw source : sink ratio
    monitor_df['source:sink_ratio'] = monitor_df['n_source'] / monitor_df['n_sink']

    # calculate the probability that the committor prediction was
    # correct for a bin

    # now calculate some bayesian statistics
    monitor_df['correct_p'] = betainc(monitor_df['n_source'] + 1,
                                      monitor_df['n_sink'] + 1, beta_upper_x) -\
                              betainc(monitor_df['n_source'] + 1,
                                      monitor_df['n_sink'] + 1, beta_lower_x)
    monitor_df['certainty'] = monitor_df['correct_p'] / (beta_upper_x - beta_lower_x)

    # send to STDOUT
    sys.stdout.write(monitor_df.__repr__() + "\n")

    if output is not None:
        monitor_df.to_csv(output)

    if outcomes_target is not None:
        outcomes_df.to_csv(outcomes_target)

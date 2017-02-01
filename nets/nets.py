import sys
import itertools as it

import click

import pandas as pd
import numpy as np
from scipy.special import betainc

from msmbuilder.dataset import dataset as msmdataset

import nets.monitor
import nets.seed

@click.group()
def cli():
    pass

@cli.command('seed')
@click.option('-O', 'output', default=None, type=click.Path(),
              help="file to write out the info table if given")
@click.option('--charmm-rst-input', 'charmm_rst_input', default=None, type=click.Path(),
              help="file to write out the input for use in a charmm script the info table if given, hopefully this will deprecate and is basically for internal protype usage only.")
@click.option('--run-traj-counts', 'run_traj_counts', default=None, type=str,
              help="the number of trajs in a run (e.g. '8,8,8,8,16') for making useful traj names a charmm script the info table if given, hopefully this will deprecate and is basically for internal protype usage only.")
@click.option('--scan', default=True,
              help="Get samples for a scan across all of the bins")
@click.option('--scan-no-ts', default=False,
              help="Get samples for a scan across all of the bins excepting the transition state")
@click.option('--ts', default=False,
              help="Get samples for the transition state bins (0.4 - 0.6)")
@click.option('--n-samples', 'n_samples', type=int,
              help="The number of seeds to sample from each bin")
@click.option('--assignments', type=click.Path(exists=True),
                help="Data-structure mapping trajectory->[cluster_id, ...] for each frame"
                "of the trajectory. This is the default dataset of MSMBuilder clustering.")
@click.argument('nodes', nargs=1, type=click.Path(exists=True))
def seed(output, charmm_rst_input, run_traj_counts, scan, scan_no_ts,
         ts, n_samples, assignments, nodes):
    assignments_path = assignments
    assignments = msmdataset(assignments_path)
    # invert the assignments mapping
    cluster_map = nets.seed.cluster_frames_map(assignments)

    nodes_df = pd.read_csv(nodes, index_col=0)
    committors = nodes_df['committor_prob'].values

    # we want only the non_basin indices
    non_basin_idxs = list(it.chain(*np.argwhere((committors != 0) & (committors != 1))))

    # make the histogram of committors
    # TODO change the bin edges, currently uses the defaults
    _, bin_edges = nets.seed.committor_hist(committors)

    # make bin tuples (l_edge, center, u_edge)
    bin_tups = nets.seed.make_bin_tuples(bin_edges)

    # assign committors to bins
    bin_assignments = nets.seed.committor_bins(committors, bin_edges)

    # choose which bins you want to actually sample
    assert not (scan and scan_no_ts and ts), "only one of [scan, scan-no-ts, ts] can be chosen"
    if scan:
        bin_idxs = range(len(bin_tups))
    elif scan_no_ts:
        bin_idxs = [bin_idx for bin_idx, bin_tup in enumerate(bin_tups) if
                    not (bin_tup[0] >= 0.4 and bin_tup[-1] <= 0.61)]
    elif ts:
        bin_idxs = [bin_idx for bin_idx, bin_tup in enumerate(bin_tups)
                    if bin_tup[0] >= 0.4 and bin_tup[-1] <= 0.61]

    # sample the frames in each bin (aggregate of all clusters in a bin)

    bin_samples = nets.seed.sample_committors(cluster_map,
                                              bin_assignments[non_basin_idxs], bin_idxs, n_samples)
    # these are called the seeds
    seeds = list(it.chain(*bin_samples.values()))

    seeds_df = nets.seed.make_seeds_df(seeds, committors,
                                       bin_assignments[non_basin_idxs], bin_tups)

    # if an output file was specified write the seeds_df as a csv there
    if output:
        seeds_df.to_csv(output)
    # or else print to STDOUT
    else:
        click.echo(seeds_df)


    # TODO
    # this part is not permanent and has to do with the way we make
    # restart files for running NETS.

    # Ideally this will be replaced by another command for generating
    # inputs but for convenience now it is here
    if charmm_rst_input:
        # from the assignments get the map to the frames to use
        traj_frames = nets.seed.get_seeds_traj_frames(cluster_map, seeds)
        # the traj_ids for making a charmm rst input file
        run_traj_counts = [int(i) for i in run_traj_counts.split(',')]
        traj_ids = nets.seed.gen_traj_ids(run_traj_counts)
        # then generate an input file to be used by a CHARMM script to make rst files
        nets.seed.gen_rst_seeds_input(traj_frames, charmm_rst_input, traj_ids)



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

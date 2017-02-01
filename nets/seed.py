import os.path as os
import itertools as it
import collections as col

import numpy as np
import pandas as pd

def cluster_frames_map(assignments):
    """Make a dict mapping cluster_idx -> (traj_idx, frame_idx) given a
    dictionary mapping traj_idx -> [frame_idx -> cluster_idx, ...].

    In short inverts an MSMBuilder cluster assignments dataset so that
    you can choose frames from a specific cluster.

    """
    cluster_map = col.defaultdict(list)
    for traj_idx, cluster_idxs in assignments.items():
        for frame_idx, cluster_idx in enumerate(cluster_idxs):
            cluster_map[cluster_idx].append((traj_idx, frame_idx))

    return cluster_map

def committor_hist(committors, n_bins=20):
    """Bin committor values given bin widths for binning across 0.0 - 1.0
    of committor probabilities.

    """
    hist, bin_edges = np.histogram(committors, bins=n_bins)

    return hist, bin_edges

def committor_bins(committors, bin_edges=np.arange(0.0, 1.05, 0.05)):
    """Assign the committor values to bins and report the bin centers."""


    # make bin assignments and decrease idxs by 1 to make 0-indexed by
    # bins
    bin_assignments = np.digitize(committors, bin_edges) - 1

    # reassign bins from sinks and source (committor 0 or 1) to -1
    bin_assignments[(committors == 0) | (committors == 1)] = -1

    return bin_assignments

def make_bin_tuples(bin_edges):
    # the bin idxs committors will be assigned to
    bin_idxs = range(len(bin_edges) - 1)

    # get the center of the bins
    bin_tups = []
    for bin_idx in bin_idxs:
        l_edge = bin_edges[bin_idx]
        u_edge = bin_edges[bin_idx + 1]
        center = (u_edge - l_edge)/2 + l_edge
        bin_tups.append((l_edge, center, u_edge))

    return bin_tups

def sample_committors(cluster_map, bin_assignments, bin_idxs, n_samples):
    """Take samples of frames from a committor bin.

    Returns:
    bin_samples : bin_idx -> (clust_idx, clust_frame_idx)

    """

    bin_dict = {bin_idx : list(it.chain(*np.argwhere(bin_assignments == bin_idx)))
                for bin_idx in bin_idxs}

    bin_samples = col.defaultdict(list)
    for bin_idx, cluster_idxs in bin_dict.items():
        clust_n_frames = [len(cluster_map[cluster_idx]) for cluster_idx in cluster_idxs]
        cum_idxs = np.cumsum(clust_n_frames)

        sample_frame_idxs = np.random.choice(cum_idxs[-1], n_samples, replace=False)
        for sample_idx in sample_frame_idxs:
            i = np.searchsorted(cum_idxs, sample_idx)
            cum_i = cum_idxs[i]
            clust_frame_idx = cum_i - sample_idx
            if clust_frame_idx == 0:
                bin_clust_idx = i + 1
            else:
                bin_clust_idx = i
            clust_idx = cluster_idxs[bin_clust_idx]
            bin_samples[bin_idx].append((clust_idx, clust_frame_idx))

    return bin_samples

def make_seeds_df(seeds, committors, bin_assignments, bin_tups):
    seeds_df = pd.DataFrame(seeds, columns=["clust_idx", "clust_frame_idx"])
    committors_col = []
    for clust_idx in seeds_df['clust_idx'].values:
        committors_col.append(committors[clust_idx])
    seeds_df['committor_prob'] = committors_col
    seed_bin_assignments = [bin_assignments[clust_idx] for clust_idx in seeds_df['clust_idx']]
    seed_bins = [bin_tups[i] for i in seed_bin_assignments]
    seeds_df['bin_idx'] = seed_bin_assignments
    seeds_df['bin_l_edge'] = [b[0] for b in seed_bins]
    seeds_df['bin_center'] = [b[1] for b in seed_bins]
    seeds_df['bin_u_edge'] = [b[2] for b in seed_bins]

    return seeds_df

def get_seeds_traj_frames(cluster_frames_map, clust_samples):
    """Given a list of (cluster_idx, clust_frame_idxs) generates a
    collection of CHARMM rst files as seeds for running NETS.

    """
    traj_frames = []
    for clust_idx, frame_idx in clust_samples:
        traj_frames.append(cluster_frames_map[clust_idx][frame_idx])

    return traj_frames

def gen_traj_ids(run_traj_counts):
    """Generates traj ids in the <run_idx>_<traj_idx> style from the
    number of trajectories in each run.

    """

    traj_ids = []
    for run_idx, n_trajs in enumerate(run_traj_counts):
        ids = ["{0:03}_{1:03}".format(run_idx + 1, run_traj_idx) for run_traj_idx
               in range(n_trajs)]
        traj_ids.extend(ids)

    return traj_ids


def gen_rst_seeds_input(traj_frame_tups, rst_seed_path, traj_ids):
    df = pd.DataFrame(traj_frame_tups, columns=['traj_idx', 'frame_idx'])
    df_gb = df.groupby('traj_idx')

    with open(rst_seed_path, 'w') as wf:
        for traj_idx, g_df in df_gb:
            traj_id = traj_ids[traj_idx]
            for frame_idx in g_df['frame_idx'].values:
                wf.write("{}\n".format(traj_id))
                wf.write("{}\n".format(frame_idx))

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
    cluster_frames_map = col.defaultdict(list)
    for traj_idx, cluster_idxs in assignments.items():
        for frame_idx, cluster_idx in enumerate(cluster_idxs):
            cluster_frames_map[cluster_idx].append((traj_idx, frame_idx))

    return cluster_frames_map

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

def sample_committors(cluster_map, bin_assignments, n_samples):
    """Take samples of frames from a committor bin.

    Returns:
    bin_samples : bin_idx -> (clust_idx, clust_frame_idx)

    """

    bin_idxs = list(range(len(bin_edges) - 1))

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
    seeds_df['committor_prob'] = committors[seeds_df['clust_idx'].values]
    seed_bin_assignments = [bin_assignments[clust_idx] for clust_idx in seeds_df['clust_idx']]
    seed_bins = [bin_tups[i] for i in seed_bin_assignments]
    seeds_df['bin_idx'] = [i for i in range(len(seed_bins))]
    seeds_df['bin_l_edge'] = [b[0] for b in seed_bins]
    seeds_df['bin_center'] = [b[1] for b in seed_bins]
    seeds_df['bin_u_edge'] = [b[2] for b in seed_bins]

    return seeds_df

def gen_rst_seeds(cluster_frames_map, clust_samples):
    """Given a list of (cluster_idx, clust_frame_idxs) generates a
    collection of CHARMM rst files as seeds for running NETS.

    """
    traj_frames = []
    for clust_idx, frame_idx in clust_samples:
        traj_frames.append(cluster_frames_map[clust_idx][frame_idx])

def monitor():
    pass

def report():
    pass

def restart():
    pass

if __name__ == "__main__":
    import msmbuilder.dataset as msmdataset
    import os.path as osp
    import pandas as pd

    seh_dir = "/home/salotz/Dropbox/lab/sEH/"
    tppu_dir = osp.join(seh_dir, "tppu_unbinding")

    dataset_path = osp.join(tppu_dir, 'clust.h5')
    assignments = msmdataset.dataset(dataset_path)
    cluster_map = cluster_frames_map(assignments)

    nodes_path = osp.join(tppu_dir, "nodes.csv")
    nodes_df = pd.read_csv(nodes_path, index_col=0)

    committors = nodes_df['committor_prob'].values
    non_basin_idxs = list(it.chain(*np.argwhere((committors != 0) & (committors != 1))))

    hist, bin_edges = committor_hist(committors)
    bin_tups = make_bin_tuples(bin_edges)

    bin_assignments = committor_bins(committors, bin_edges)
    # take out the basins
    bin_samples = sample_committors(cluster_map, bin_assignments[non_basin_idxs], 5)

    seeds = list(it.chain(*bin_samples.values()))

    seeds_df = make_seeds_df(seeds, committors, bin_assignments[non_basin_idxs], bin_tups)

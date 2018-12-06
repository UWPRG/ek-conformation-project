import itertools
import numpy as np
import pandas as pd
import mdtraj as md

import matplotlib.pyplot as plt

import sys
sys.path.append("..")
from utilities import read_plumed_file, reweight_ct

#############
# load traj #
#############

# simulation files to read
temps = [
    300.000, 311.264, 322.952, 335.078, 347.660, 360.714,
    374.258, 388.311, 402.891, 418.019, 433.715, 450.000
]
temp_idx = 0
pdb = '/Users/joshsmith/Git/ek-conformation-project/analysis/clustering/EKGG.pdb'
xtc = f'/Volumes/UntitledUnmastered/EK_proj/EKGG/MetaD/centered_{temp_idx}.xtc'
# ct file
ct_file = f'/Volumes/UntitledUnmastered/EK_proj/EKGG/' \
          f'driver/Reweighting/temp{temp_idx}/COLVARdriver'

time_range = (0, 300000)  # ps


# load low temp traj and restrict frames to time range of interest
traj = md.load(xtc, top=pdb, stride=100)
traj = traj[np.where(traj.time >= time_range[0])]
traj = traj[np.where(traj.time < time_range[1])]


#########################
# load rewieghting bias #
#########################

# Read in colvar for weight
ct = read_plumed_file(ct_file, bias_column='metad.rbias')
ct = ct[traj.time]

frame_weights = pd.DataFrame(reweight_ct(ct, temp=temps[temp_idx]), index=ct.index)
frame_weights['norm_wt'] = frame_weights / frame_weights.sum()


######################
# calculate contacts #
######################
top = traj.topology
# define possible salt bridges
backbone_idx = top.select("backbone and name CA")
# calculate pairwise backbone rmsd
distances = np.empty((traj.n_frames, traj.n_frames))
traj.center_coordinates()
# could be parallelized with multiprocess in python!
for i in range(traj.n_frames):
    distances[i] = md.rmsd(
        traj,
        traj,
        frame=i,
        atom_indices=backbone_idx,
        precentered=True,
    )
print('Max pairwise rmsd: %f nm' % np.max(distances))


def gromos(rmsd_matrix, cutoff):
    """
    Calculate clusters via the 'gromos' method proposed by Daura et al.

    Parameters
    ----------
    rmsd_matrix :
        Square matrix of
    cutoff :
        RMSD cutoff for "neighbors", in nm.

    Returns
    -------
    clust_ids :
    """
    rmsd_matrix = np.copy(rmsd_matrix)
    clust_ids = np.zeros(rmsd_matrix.shape[0])

    clust_idx = 1
    while not np.all(np.isnan(rmsd_matrix)):
        # identify structure with the most neighbors
        neighbor_matrix = (rmsd_matrix < cutoff)
        most_neighborly_cluster = np.argmax(neighbor_matrix.sum(axis=0))
        new_cluster_member_ids = np.where(neighbor_matrix[most_neighborly_cluster])[0]
        # stop clustering once there are no clusters greater than 1 structure in size
        if len(new_cluster_member_ids) == 1:
            break
        # assign cluster id to each member of the current cluster
        clust_ids[new_cluster_member_ids] = clust_idx
        # remove members of current cluster from consideration for future clusters
        rmsd_matrix[new_cluster_member_ids, :] = np.nan
        rmsd_matrix[:, new_cluster_member_ids] = np.nan
        clust_idx += 1

    return clust_ids


clust_ids = pd.Series(
    gromos(distances, 0.6),
    index=traj.time,
)

frame_weights['clust_id'] = clust_ids

clust_wts = frame_weights.groupby(['clust_id'])[['norm_wt']].sum()


print('lol')
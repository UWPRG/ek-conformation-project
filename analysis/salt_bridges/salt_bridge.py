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
sequences = [
    'EK', 'EKG', 'EKGG', 'EGKG', 'EKGGG', 'GG',
]
temps = [
    300.000, 311.264, 322.952, 335.078, 347.660, 360.714,
    374.258, 388.311, 402.891, 418.019, 433.715, 450.000
]

seq_idx = 3

for temp_idx, temp in enumerate(temps):
    # topology
    pdb = f'/Users/joshsmith/Git/ek-conformation-project/analysis/clustering/' \
          f'{sequences[seq_idx]}.pdb'
    # CENTERED (!!) trajectory
    xtc = f'/Volumes/UntitledUnmastered/EK_proj/{sequences[seq_idx]}/MetaD/' \
          f'centered_{temp_idx}.xtc'
    # ct file
    ct_file = f'/Volumes/UntitledUnmastered/EK_proj/{sequences[seq_idx]}/' \
              f'driver/Reweighting/temp{temp_idx}/COLVARdriver'

    time_range = (0, 300000)  # ps

    # load low temp traj and restrict frames to time range of interest
    traj = md.load(xtc, top=pdb)
    traj = traj[np.where(traj.time >= time_range[0])]
    traj = traj[np.where(traj.time < time_range[1])]


    #########################
    # load rewieghting bias #
    #########################

    # Read in colvar for weight
    ct = read_plumed_file(ct_file, bias_column='metad.rbias')  #, stride=100)
    ct = ct[traj.time]

    frame_weights = pd.DataFrame(reweight_ct(ct, temp=temp), index=ct.index)
    frame_weights['norm_wt'] = frame_weights / frame_weights.sum()

    ######################
    # calculate contacts #
    ######################
    top = traj.topology
    # define possible salt bridges
    pos_charge_center = top.select("resname LYS and name NZ")
    neg_charge_center = top.select("resname GLU and name CD")
    bridges = np.array(
        list(itertools.product(pos_charge_center, neg_charge_center))
    )
    # caluclate distances between charge centers in all frames
    charge_dists = pd.DataFrame(
        md.compute_distances(traj, atom_pairs=bridges, periodic=False),
        index=traj.time,
    )
    # calculate number of salt bridges in eah frame
    dist_cutoff = 0.4
    salt_bridge_count = (charge_dists < dist_cutoff).sum(axis=1)
    # weight by normalized frame weight
    avg_num_salt_bridges = sum(frame_weights['norm_wt'] * salt_bridge_count)


    #####################
    # calculate entropy #
    #####################
    phi = md.compute_phi(traj)
    psi = md.compute_psi(traj)

    ramachandrans = []
    for idx in range(phi[1].shape[1] - 1):
        x = phi[1][:, idx]
        y = psi[1][:, idx + 1]
        hist, _, _ = np.histogram2d(
            x, y, bins=45, weights=frame_weights['norm_wt'].values, density=True,
        )
        s_j = -hist * np.ma.log(hist)
        ramachandrans.append(s_j.sum())
    conf_entropy = np.array(ramachandrans).mean()


    ###########
    # do dssp #
    ###########

    structure = md.compute_dssp(traj)
    structure[structure == "H"] = 0.0
    structure[structure == "E"] = 1.0
    structure[structure == "C"] = 2.0
    structure = np.array(structure, dtype=float)
    plt.imshow(structure.T[:, ::500])
    plt.show()

    with open(f'temp_compare_{sequences[seq_idx]}.txt', mode='a') as f:
        print(temp, avg_num_salt_bridges, conf_entropy, file=f)
    # print(hi_traj)

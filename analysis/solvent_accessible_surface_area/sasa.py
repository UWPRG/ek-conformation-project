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
temp_idx = 0
for seq_idx in range(5, 6):
    pdb = f'/Users/joshsmith/Git/ek-conformation-project/analysis/clustering/{sequences[seq_idx]}.pdb'
    xtc = f'/Volumes/UntitledUnmastered/EK_proj/{sequences[seq_idx]}/MetaD/centered_{temp_idx}.xtc'
    # ct file
    ct_file = f'/Volumes/UntitledUnmastered/EK_proj/{sequences[seq_idx]}/' \
              f'driver/Reweighting/temp{temp_idx}/COLVARdriver'

    time_range = (0, 300000)  # ps

    # load low temp traj and restrict frames to time range of interest
    traj = md.load(xtc, top=pdb, stride=10)
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

    ##################
    # calculate sasa #
    ##################

    sasa = md.shrake_rupley(traj)
    hydrophilic_selection = "name O N H OE2 OE1 NZ HZ1 HZ2 HZ3"  # end cap H1 H2 H3 OC1 OC2
    hydrophilic_idx = traj.top.select(hydrophilic_selection)

    hydrophilic_sasa = pd.DataFrame(
        sasa[:, hydrophilic_idx].sum(axis=1),
        index=traj.time,
    )
    total_sasa = pd.DataFrame(
        sasa.sum(axis=1),
        index=traj.time,
    )
    # calculate avg
    avg_hydrophilic_sasa = (
        hydrophilic_sasa[0] * frame_weights['norm_wt']
    ).sum()
    avg_total_sasa = (
            total_sasa[0] * frame_weights['norm_wt']
    ).sum()




    ######################
    # calculate contacts #
    ######################
    # top = traj.topology
    # # define possible salt bridges
    # pos_charge_center = top.select("resname LYS and name NZ")
    # neg_charge_center = top.select("resname GLU and name CD")
    # bridges = np.array(
    #     list(itertools.product(pos_charge_center, neg_charge_center))
    # )
    # # caluclate distances between charge centers in all frames
    # charge_dists = pd.DataFrame(
    #     md.compute_distances(traj, atom_pairs=bridges, periodic=False),
    #     index=traj.time,
    # )
    # # calculate number of salt bridges in eah frame
    # dist_cutoff = 0.4
    # salt_bridge_count = (charge_dists < dist_cutoff).sum(axis=1)
    salt_bridge_count = np.zeros_like(frame_weights['norm_wt'])

    print("salt_bridges total_sasa t_std hydrophilic_sasa h_std")
    # for count in range(salt_bridge_count.max()):
    #     frame_id = np.where(salt_bridge_count == count)

    hydrophilic_frxn = (
            hydrophilic_sasa.values[:]
            / total_sasa.values[:]
    )
    print(
        0,
        hydrophilic_sasa.values[:].mean(),
        hydrophilic_sasa.values[:].std(),
        total_sasa.values[:].mean(),
        total_sasa.values[:].std(),
        hydrophilic_frxn.mean(),
        hydrophilic_frxn.std(),
    )

    print(sequences[seq_idx], avg_hydrophilic_sasa)

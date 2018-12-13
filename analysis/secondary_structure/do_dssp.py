import itertools
import os.path as op
import numpy as np
import pandas as pd
import mdtraj as md
import matplotlib.pyplot as plt

import sys
sys.path.append("..")
from utilities import read_plumed_file, reweight_ct
from structure import calculate_contacts, do_dssp

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

# select sequence and temperature
# seq_idx = 4
# temp_idx = 0

pdb_dir = '/Users/joshsmith/Git/ek-conformation-project/analysis/clustering/'
# xtc_dir = '/Volumes/UntitledUnmastered/EK_proj/'
xtc_dir = '/Users/joshsmith/Git/ek-conformation-project/analysis/practice_data/'
plumed_dir = '/Users/joshsmith/Git/ek-conformation-project/analysis/practice_data/'

time_range = (0, 300000)  # ps

for seq_idx in range(1):
    for temp_idx, temp in enumerate(temps):
        # topology
        pdb = op.join(pdb_dir, f'{sequences[seq_idx]}.pdb')
        # trajectory
        # xtc = op.join(xtc_dir, f'{sequences[seq_idx]}/MetaD/centered_{temp_idx}.xtc')
        xtc = op.join(xtc_dir, 'EK_300_skip_100.xtc')

        # load low temp traj and restrict frames to time range of interest
        traj = md.load(xtc, top=pdb)
        traj = traj[np.where(traj.time >= time_range[0])]
        traj = traj[np.where(traj.time < time_range[1])]


        #########################
        # load rewieghting bias #
        #########################
        # ct file
        # ct_file = f'/Volumes/UntitledUnmastered/EK_proj/{sequences[seq_idx]}/' \
        #           f'driver/Reweighting/temp{temp_idx}/COLVARdriver'
        ct_file = op.join(plumed_dir, 'EK_300_COLVAR_driver_skip_400')

        # Read in colvar for weight
        ct = read_plumed_file(ct_file, bias_column='metad.rbias')
        ct = ct[traj.time]

        frame_weights = pd.DataFrame(
            reweight_ct(ct, temp=temps[temp_idx]),
            index=ct.index
        )
        frame_weights['norm_wt'] = frame_weights / frame_weights.sum()

        ######################
        # calculate contacts #
        ######################

        # define possible salt bridges
        pos_charge_center = "resname LYS and name NZ"
        neg_charge_center = "resname GLU and name CD"

        is_salt_bridge, atom_pairs = calculate_contacts(
            traj, pos_charge_center, neg_charge_center, cutoff=0.4
        )


        ###########
        # do dssp #
        ###########
        #
        # structure = pd.DataFrame(
        #     md.compute_dssp(traj),
        #     index=traj.time,
        # )
        #
        # helical_mask = (structure == "H")  # for fraction, add: .mean(axis=1)
        # helix_frxn = (helical_mask.mean(axis=1) * frame_weights['norm_wt']).sum()
        # extended_mask = (structure == "E")  # for fraction, add: .mean(axis=1)
        # extended_frxn = (extended_mask.mean(axis=1) * frame_weights['norm_wt']).sum()
        # coil_mask = (structure == "C")  # for fraction, add: .mean(axis=1)
        # coil_frxn = (coil_mask.mean(axis=1) * frame_weights['norm_wt']).sum()
        # # residue-wise salt bridge + structure
        # salt_bridge_mask = pd.DataFrame(
        #     np.zeros_like(structure, dtype=bool),
        #     index=traj.time,
        # )
        # # for each frame, set True where salt bridge occurs
        # for t in is_salt_bridge.index:
        #     # TODO: vectorize once i figure out how to slice mdtraj topology
        #     for atom_idx in bridges[np.where(is_salt_bridge.loc[t, :])]:
        #         for atom in atom_idx:
        #             res_id = top.atom(atom).residue.index
        #             salt_bridge_mask.loc[t, res_id] = True
        # # find where salt bridge AND given structure is true
        # helix_and_salt_bridge = (helical_mask * salt_bridge_mask).sum(axis=1)
        # extended_and_salt_bridge = (extended_mask * salt_bridge_mask).sum(axis=1)
        # coil_and_salt_bridge = (coil_mask * salt_bridge_mask).sum(axis=1)
        # # filter to only helical frames and reweight fraction
        # nonzero_helix_frames = helical_mask.sum(axis=1).nonzero()
        # helix_salt_bridge_frxn = (
        #     (
        #         helix_and_salt_bridge.iloc[nonzero_helix_frames]
        #         / helical_mask.sum(axis=1).iloc[nonzero_helix_frames]
        #     ) * frame_weights['norm_wt']
        # ).sum()
        #
        # nonzero_extended_frames = extended_mask.sum(axis=1).nonzero()
        # extended_salt_bridge_frxn = (
        #     (
        #         extended_and_salt_bridge.iloc[nonzero_extended_frames]
        #         / extended_mask.sum(axis=1).iloc[nonzero_extended_frames]
        #     ) * frame_weights['norm_wt']
        # ).sum()
        #
        # nonzero_coil_frames = coil_mask.sum(axis=1).nonzero()
        # coil_salt_bridge_frxn = (
        #     (
        #         coil_and_salt_bridge.iloc[nonzero_coil_frames]
        #         / coil_mask.sum(axis=1).iloc[nonzero_coil_frames]
        #     ) * frame_weights['norm_wt']
        # ).sum()

        # ###########
        # # do dssp #
        # ###########

        structure_frxn = do_dssp(traj, simplified=False)
        # get reweighted structure frxn by broadcasting normalized frame weights
        reweighted_frxn = structure_frxn.multiply(frame_weights['norm_wt'], axis='index')


        # structure[structure == "H"] = 0.0
        # structure[structure == "E"] = 1.0
        # structure[structure == "C"] = 2.0
        # structure = np.array(structure, dtype=float)
        # plt.imshow(structure.T[:, ::500])
        # plt.show()

        with open(f'full_struct_{sequences[seq_idx]}.txt', mode='a') as f:
            print(
                temp,
                alpha_frxn,
                three_ten_frxn,
                pi_frxn,
                bridge_frxn,
                extended_frxn,
                turn_frxn,
                bend_frxn,
                coil_frxn,
                file=f
            )

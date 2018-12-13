import os.path as op
import numpy as np
import pandas as pd
import mdtraj as md

import sys
sys.path.append("..")
from utilities import read_plumed_file, reweight_ct
from contacts import calculate_contacts

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
            traj, pos_charge_center, neg_charge_center, cutoff=0.4,
        )
        salt_bridge_count = is_salt_bridge.sum(axis=1)
        # get reweighted structure frxn by broadcasting normalized frame weights
        avg_num_salt_bridges = salt_bridge_count.multiply(
            frame_weights['norm_wt'], axis='index'
        ).sum()

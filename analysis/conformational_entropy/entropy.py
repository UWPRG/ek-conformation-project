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
pdb = '/Users/joshsmith/Git/ek-conformation-project/analysis/clustering/clust1.pdb'
xtc = f'/Volumes/UntitledUnmastered/EK_proj/EK/MetaD/traj_comp{temp_idx}.xtc'
# ct file
ct_file = f'/Volumes/UntitledUnmastered/EK_proj/EK/' \
          f'driver/Reweighting/temp{temp_idx}/COLVARdriver'

time_range = (150000, 300000)  # ps

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

frame_weights = pd.DataFrame(reweight_ct(ct, temp=temps[temp_idx]), index=ct.index)
frame_weights['norm_wt'] = frame_weights / frame_weights.sum()

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

with open('temp_compare_not_periodic_short_cut.txt', mode='a') as f:
    print(temps[temp_idx], conf_entropy, file=f)


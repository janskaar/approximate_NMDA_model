"""
Extract data from the decision making simulations to save time when copying from HPC
"""

import os, h5py, pickle
import numpy as np
from pathlib import Path

# toggle for the two different simulations, other at the end of script
# data_dir = os.path.join(Path(__file__).parent, f"decision_making_sparse")
data_dir = os.path.join(Path(__file__).parent, f"decision_making_sparse_adjusted")

def load_file(d, key):
    outer_vals = []
    for i in range(1, 9, 1):
        inner_vals = []
        with h5py.File(os.path.join(d, f"runner_{i}.h5"), "r") as f:
            grps = sorted(list(f.keys()), key=int)
            for grp in grps:
                inner_vals.append(f[grp][key][()])
        outer_vals.append(inner_vals)
    return np.array(outer_vals)

keys = [
        "s_NMDA_pre_selective1_mean",
        "s_NMDA_pre_selective2_mean",
        "s_NMDA_selective1_mean",
        "s_NMDA_selective2_mean",

        "I_NMDA_selective1_mean",
        "I_NMDA_selective2_mean",

        "I_GABA_selective1_mean",
        "I_GABA_selective2_mean",

        "I_AMPA_selective1_mean",
        "I_AMPA_selective2_mean",

        "hist_selective1",
        "hist_selective2",
        ]

keys_short = [
        "s_NMDA_pre1",
        "s_NMDA_pre2",
        "s_NMDA1",
        "s_NMDA2",

        "I_NMDA1",
        "I_NMDA2",

        "I_GABA1",
        "I_GABA2",

        "I_AMPA1",
        "I_AMPA2",

        "hist1",
        "hist2",
        ]

res = {}
for i, key in enumerate(keys):
    vals = load_file(data_dir, key)
    vals = np.stack([v.reshape((-1, *v.shape[2:])) for v in vals])
    res[keys_short[i]] = vals.astype(np.float32)
 
# toggle for the two different simulations
# with open("decision_making_sparse.pkl", "wb") as f:
with open("decision_making_sparse_adjusted.pkl", "wb") as f:
    pickle.dump(res, f)


import mne
from compensate import compensate
import numpy as np
import pickle
from scipy.spatial.distance import directed_hausdorff as d_hd


subjs = ["ATT_10","ATT_11","ATT_12","ATT_13","ATT_14"]
runs = ["1","2","3","4","5"]
#subjs = ["ATT_24"]
#runs = ["3"]

proc_dir = "/media/hdd/jeff/reftest/proc/"

with open("/media/hdd/jeff/reftest/bin/compsup1","rb") as f:
    weights = pickle.load(f)
weights = weights["digital"]

dists = []
for sub in subjs:
    for run in runs:
        raw = mne.io.Raw("{dir}nc_{sub}_{run}_sim-raw.fif".format(
                         dir=proc_dir,sub=sub,run=run),preload=True)
        raw.info["bads"].remove("MRyA")
        raw.info["bads"].remove("MRyaA")
        compensate(raw,weights,direction=-1)
        estimator = compensate(raw)
        coef = estimator.coef_
        meg_picks = mne.pick_types(raw.info,meg=True,ref_meg=False)
        ref_picks = mne.pick_channels(raw.ch_names,weights["comp_names"])
        meg_ch_names = [raw.ch_names[x] for x in meg_picks]
        ref_ch_names = [raw.ch_names[x] for x in ref_picks]

        fixed_coef = np.empty(coef.shape)
        tempweigh = weights["weights"]
        ref_key = weights["comp_names"]
        for ch_idx,ch in enumerate(meg_ch_names):
            for rch_idx,rch in enumerate(ref_ch_names):
                fixed_coef[ch_idx,rch_idx] = tempweigh[ch][rch_idx]

        dists.append(d_hd(fixed_coef,coef)[0])

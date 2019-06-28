import mne
import numpy as np
import pickle
from compensate import compensate

raw_dir ="/home/jeff/ATT_dat/proc/"
proc_dir = "/home/jeff/reftest/proc/"
subjs = ["ATT_10","ATT_11","ATT_12","ATT_13","ATT_14"]
runs = ["2","3","4","5"]

with open("/home/jeff/reftest/bin/compsup1","rb") as f:
    compsup1 = pickle.load(f)

for sub in subjs:
    for run in runs:
        workfile = "{dir}nc_{s}_{r}_hand-raw.fif".format(dir=raw_dir,s=sub,r=run)
        raw = mne.io.Raw(workfile)
        compensate(raw,weights=compsup1["digital"],direction=-1)
        compensate(raw)
        picks = mne.pick_types(raw.info,meg=True,ref_meg=True) # get channels we want to filter
        raw.save("{dir}{sub}_{run}-raw.fif".format(dir=proc_dir,sub=sub,run=run),overwrite=True)

        ica = mne.preprocessing.ICA(n_components=0.999,allow_ref_meg=True,
                                    max_iter=10000,method="picard")
        # ica.fit(raw,picks=picks)
        # ica.save("{dir}{sub}_{run}-ica.fif".format(dir=proc_dir,sub=sub,run=run))

        picks = mne.pick_types(raw.info,meg=False,ref_meg=True)
        ica = mne.preprocessing.ICA(n_components=0.99,allow_ref_meg=True,
                                    max_iter=10000,method="picard")
        ica.fit(raw,picks=picks)
        ica.save("{dir}{sub}_{run}_ref-ica.fif".format(dir=proc_dir,sub=sub,run=run))

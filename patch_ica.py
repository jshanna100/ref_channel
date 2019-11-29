import mne
from sklearn.linear_model import LinearRegression
import numpy as np
import pickle
from os import walk

proc_dir = "/home/jeff/reftest/proc/"
files = list(walk(proc_dir))[0][2]
subjs = ["ATT_10","ATT_11","ATT_12","ATT_13","ATT_14"]
runs = ["2","3","4","5"]
ica_ranges = [20,40,60,80,100]

filebases = []
for sub in subjs:
    for run in runs:
        for n_idx in range(100):
            filebases.append("nc_{sub}_{run}_{n}".format(sub=sub,run=run,
                                                         n=n_idx))

for f_idx,filebase in enumerate(filebases):
    these_files = [filebase+"_ref-ica.fif"]
    for c_num in ica_ranges:
        these_files.append("{}_c{}_meg-ica.fif".format(filebase,c_num))
        these_files.append("{}_c{}-ica.fif".format(filebase,c_num))
    if not all([x in files for x in these_files]):
        filename = proc_dir+filebase+"_sim-raw.fif"
        with open(proc_dir+filebase+".est","rb") as f:
            estimator = pickle.load(f)
        raw = mne.io.Raw(filename,preload=True)
        ref_picks = mne.pick_types(raw.info,meg=False,ref_meg=True)
        meg_picks = mne.pick_types(raw.info,meg=True,ref_meg=False)
        Y_pred = estimator.predict(raw[ref_picks][0].T)
        oldraw = raw.copy()
        raw._data[meg_picks] -= Y_pred.T

        if these_files[0] not in files:
            icaref = mne.preprocessing.ICA(n_components=6,max_iter=1000,
                                       method="picard",allow_ref_meg=True)
            picks = mne.pick_types(raw.info,meg=False,ref_meg=True)
            icaref.fit(raw,picks=picks)
            icaref.save(proc_dir+filebase+"_ref-ica.fif")

        for c_num in ica_ranges:
            if filebase+"_c{}_meg-ica.fif".format(c_num) not in files:
                icameg = mne.preprocessing.ICA(n_components=c_num,max_iter=1000,
                                               method="picard")
                picks = mne.pick_types(raw.info,meg=True,ref_meg=False)
                icameg.fit(raw,picks=picks)
                icameg.save(proc_dir+filebase+"_c{}_meg-ica.fif".format(c_num))
            if filebase+"_c{}-ica.fif".format(c_num) not in files:
                ica = mne.preprocessing.ICA(n_components=c_num,max_iter=1000,
                                            method="picard",allow_ref_meg=True)
                picks = mne.pick_types(raw.info,meg=True,ref_meg=True)
                ica.fit(raw,picks=picks)
                ica.save(proc_dir+filebase+"_c{}-ica.fif".format(c_num))

    else:
        print("All files present; skipping")

import mne
import numpy as np
from mne.preprocessing.ica import _ica_explained_variance

subjs = ["ATT_10"]
runs = ["1"]
proc_dir = "/home/jeff/reftest/proc/"

for sub in subjs:
    for run in runs:
        raw = mne.io.Raw("{dir}nc_{sub}_{run}_hand_ica-raw.fif".format(
                         dir=proc_dir,sub=sub,run=run),preload=True)
        ica = mne.preprocessing.ICA(n_components=200,method="picard",max_iter=3000)
        ica.fit(raw)

        raw_s = raw.copy()
        raw_s.apply_function(np.random.permutation)
        ica_s = mne.preprocessing.ICA(n_components=200,method="picard",max_iter=3000)
        ica_s.fit(raw_s)

        spurious_percent = _ica_explained_variance(ica_s,raw_s)[0]
        print(spurious_percent)

        comps_to_throw = np.where(_ica_explained_variance(ica,raw)>spurious_percent)
        raw_noise = raw.copy()
        raw_noise = ica.apply(raw,exclude=list(comps_to_throw[0]))
        cov = mne.compute_raw_covariance(raw_noise)
        mne.write_cov("{dir}nc_{sub}_{run}-cov.fif".format(
                         dir=proc_dir,sub=sub,run=run),cov)

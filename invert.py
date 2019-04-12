import mne
import numpy as np

subjs = ["ATT_10"]
runs = ["1"]
proc_dir = "/home/jeff/reftest/proc/"

for sub in subjs:
    for run in runs:
        raw = mne.io.Raw("{dir}nc_{sub}_{run}_hand_ica-raw.fif".format(
                         dir=proc_dir,sub=sub,run=run),preload=True)
        cov = mne.read_cov("{dir}empty-cov.fif".format(
                         dir=proc_dir,sub=sub,run=run))
        fwd = mne.read_forward_solution("{dir}nc_{sub}_{run}-fwd.fif".format(
                         dir=proc_dir,sub=sub,run=run))
        #fwd = mne.convert_forward_solution(fwd,force_fixed=True)
        inv = mne.minimum_norm.make_inverse_operator(raw.info,fwd,cov,fixed=True)
        stc = mne.minimum_norm.apply_inverse_raw(raw,inv,1,buffer_size=10000,method="MNE")
        stc.save("{dir}nc_{sub}_{run}".format(dir=proc_dir,sub=sub,run=run))

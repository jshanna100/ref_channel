import mne
import numpy as np

subjs = ["ATT_10"]
runs = ["1"]
proc_dir = "/home/jeff/reftest/proc/"
raw_dir = "/home/jeff/reftest/raw/emptyroom/"

convert = False

notches = [50, 62, 100, 150, 200]
breadths = np.array([1.5, 0.5, 0.5, 0.5, 0.5])

for sub in subjs:
    for run in runs:
        if convert:
            raw = mne.io.read_raw_bti("{dir}c,rfhp1.0Hz".format(dir=raw_dir),
                                     preload=True, rename_channels=False,
                                     head_shape_fname=None,
                                     config_fname="{dir}config".format(dir=raw_dir))
            picks = mne.pick_types(raw.info,meg=True,ref_meg=True)
            raw.notch_filter(notches,n_jobs="cuda",picks=picks, notch_widths=breadths)
            raw.save("{dir}empty-raw.fif".format(dir=proc_dir),overwrite=True)
        else:
            raw = mne.io.Raw("{dir}empty-raw.fif".format(proc_dir),preload=True)
            picks = mne.pick_types(raw.info,meg=True,ref_meg=True)

        cov = mne.compute_raw_covariance(raw)
        mne.write_cov("{dir}empty-cov.fif".format(dir=proc_dir),cov)
        cov_ref = mne.compute_raw_covariance(raw, picks=picks)
        mne.write_cov("{dir}empty_ref-cov.fif".format(dir=proc_dir),cov_ref)

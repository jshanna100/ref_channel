import mne
import numpy as np

# Convert from BTI format to MNE-Python

base_dir ="/home/jeff/reftest/"
raw_dir = base_dir+"raw/"
proc_dir = base_dir+"proc/"

l_freq=None
h_freq=None
notches = [50, 62, 100, 150, 200]
breadths = np.array([1.5, 0.5, 0.5, 0.5, 0.5])

subjs = ["zw","w"]
runs = ["first","second"]

for sub in subjs:
    for run in runs:
        workfile = "{dir}{s}/{r}/c,rfhp1.0Hz".format(dir=raw_dir,s=sub,r=run)
        raw = mne.io.read_raw_bti(workfile,preload=True, head_shape_fname=None,
                                     rename_channels=False)
        picks = mne.pick_types(raw.info,meg=True,ref_meg=True) # get channels we want to filter
        raw.filter(l_freq,h_freq,picks=picks,n_jobs="cuda")
        raw.notch_filter(notches,n_jobs="cuda",picks=picks, notch_widths=breadths)
        raw = raw.resample(200,n_jobs="cuda")
        raw.save("{dir}{sub}_{run}-raw.fif".format(dir=proc_dir,sub=sub,run=run))

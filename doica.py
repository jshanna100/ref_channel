import mne
from compensate import compensate

base_dir ="/media/hdd/jeff/reftest/"
proc_dir = base_dir+"proc/"
subjs = ["ATT_10","ATT_11","ATT_12","ATT_13","ATT_14"]
runs = ["1","2","3","4","5"]

for sub in subjs:
    for run in runs:
        for n_idx in range(40):
            filename = "{dir}nc_{sub}_{run}_{n}_sim-raw.fif".format(dir=proc_dir,sub=sub,run=run,n=n_idx)
            raw = mne.io.Raw(filename)
            compensate(raw)
            picks = mne.pick_types(raw.info,meg=True,ref_meg=True)
            ica = mne.preprocessing.ICA(n_components=200,allow_ref_meg=True,max_iter=10000,method="picard")
            ica.fit(raw)
            ica.save("{dir}nc_{sub}_{run}_{n}-_sim_ica.fif".format(dir=proc_dir,sub=sub,run=run,n=n_idx))

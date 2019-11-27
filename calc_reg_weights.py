import mne
from compensate import compensate
import pickle

base_dir ="/media/hdd/jeff/reftest/"
proc_dir = base_dir+"proc/"
subjs = ["ATT_10","ATT_11","ATT_12","ATT_13","ATT_14"]
runs = ["2","3","4","5"]

for sub in subjs:
    for run in runs:
        for n_idx in range(100):
            filename = "{dir}nc_{sub}_{run}_{n}_outsim-raw.fif".format(
                        dir=proc_dir,sub=sub,run=run,n=n_idx)
            raw = mne.io.Raw(filename)
            estimator = compensate(raw)
            with open("{dir}nc_{sub}_{run}_{n}.est".format(
                      dir=proc_dir,sub=sub,run=run,n=n_idx),"wb") as f:
                pickle.dump(estimator,f)

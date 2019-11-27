import mne
import argparse
from sklearn.linear_model import LinearRegression
import numpy as np
import pickle

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

parser = argparse.ArgumentParser()
parser.add_argument('--chunk', type=int, required=True)
opt = parser.parse_args()

proc_dir = "/home/woody/mfnc/mfnc001h/sims"
subjs = ["ATT_10","ATT_11","ATT_12","ATT_13","ATT_14"]
runs = ["2","3","4","5"]
filebases = []
for sub in subjs:
    for run in runs:
        for n_idx in range(100):
            filebases.append("{dir}nc_{sub}_{run}_{n}".format(dir=proc_dir,
                                                              sub=sub,run=run,
                                                              n=n_idx))
filebases = list(chunks(filebases,25))
ica_ranges = [20,40,60]

for f_idx,filebase in enumerate(filebases[opt.chunk]):
    filename = filebase+"_sim-raw.fif"
    with open(filebase+".est","rb") as f:
        estimator = pickle.load(f)
    raw = mne.io.Raw(filename,preload=True)
    ref_picks = mne.pick_types(raw.info,meg=False,ref_meg=True)
    meg_picks = mne.pick_types(raw.info,meg=True,ref_meg=False)
    Y_pred = estimator.predict(raw[ref_picks][0].T)
    oldraw = raw.copy()
    raw._data[meg_picks] -= Y_pred.T

    icaref = mne.preprocessing.ICA(n_components=6,max_iter=1000,
                               method="picard",allow_ref_meg=True)
    picks = mne.pick_types(raw.info,meg=False,ref_meg=True)
    icaref.fit(raw,picks=picks)
    icaref.save(filebase+"_ref-ica.fif")

    for c_num in ica_ranges:
        icameg = mne.preprocessing.ICA(n_components=c_num,max_iter=1000,
                                       method="picard")
        picks = mne.pick_types(raw.info,meg=True,ref_meg=False)
        icameg.fit(raw,picks=picks)
        icameg.save(filebase+"_c{}_meg-ica.fif".format(c_num))
        ica = mne.preprocessing.ICA(n_components=c_num,max_iter=1000,
                                    method="picard",allow_ref_meg=True)
        picks = mne.pick_types(raw.info,meg=True,ref_meg=True)
        ica.fit(raw,picks=picks)
        ica.save(filebase+"_c{}-ica.fif".format(c_num))

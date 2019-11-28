import mne
import pickle
import numpy as np
from compensate import compensate
import time
from scipy.stats import pearsonr
from mne.preprocessing.bads import find_outliers
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--compno', type=int, default=None)
opt = parser.parse_args()

proc_dir = "/home/jeff/reftest/proc/"
proc_dir = "/home/woody/mfnc/mfnc001h/sims/"
subjs = ["ATT_10","ATT_11","ATT_12","ATT_13","ATT_14"]
runs = ["2","3","4","5"]
# subjs = ["ATT_11"]
# runs = ["3"]
n_num = (0,100)
threshes = [.2,.3,.4,.5,.6,.7,.8,.9]
z_threshes = [2.5,3,3.5,4]
gnd_thresh = 3
separate = True
comp_nums = [20,40,60,80,100]
if opt.compno:
    comp_nums = [opt.compno]
if not separate:
    threshes=z_threshes
threshes=z_threshes
for cn in comp_nums:
    for thresh in threshes:
        threshold = thresh
        hits = {"rr":[],"nn":[],"src_inds":[],"subj_run":[],"comp":[],"cor":[]}
        misses = {"rr":[],"nn":[],"src_inds":[],"subj_run":[],"comp":[],"cor":[]}
        false_alarms = {"rr":[],"nn":[],"src_inds":[],"subj_run":[],"comp":[],"cor":[]}
        silents = {"rr":[],"nn":[],"src_inds":[],"subj_run":[],"cor":[]}
        for sub in subjs:
            for run in runs:
                for n_idx in range(n_num[0],n_num[1]):
                    with open("{dir}const_{n}".format(dir=proc_dir,n=n_idx),'rb') as f:
                        constellation = pickle.load(f)
                    x2, y2, z2 = constellation["pos"]["rr"].T
                    signal = constellation["signal"]
                    raw = mne.io.Raw("{dir}nc_{sub}_{run}_{n}_sim-raw.fif".format(dir=proc_dir,sub=sub,run=run,n=n_idx),preload=True)
                    with open("{dir}nc_{sub}_{run}_{n}.est".format(dir=proc_dir,sub=sub,run=run,n=n_idx),"rb") as f:
                        estimator = pickle.load(f)
                    ref_picks = mne.pick_types(raw.info,meg=False,ref_meg=True)
                    meg_picks = mne.pick_types(raw.info,meg=True,ref_meg=False)
                    Y_pred = estimator.predict(raw[ref_picks][0].T)
                    raw._data[meg_picks] -= Y_pred.T
                    ica = mne.preprocessing.read_ica("{dir}nc_{sub}_{run}_{n}_c{cn}-ica.fif".format(dir=proc_dir,sub=sub,run=run,n=n_idx,cn=cn))
                    if separate:
                        ref_ica = mne.preprocessing.read_ica("{dir}nc_{sub}_{run}_{n}_ref-ica.fif".format(dir=proc_dir,sub=sub,run=run,n=n_idx))
                        ref_src = ref_ica.get_sources(raw)
                        for ch_idx,ch in enumerate(ref_src.ch_names):
                            ref_src.rename_channels({ch:"REF_ICA"+str(ch_idx)})
                        raw_s = raw.copy()
                        raw_s.add_channels([ref_src])
                        inds,scores = ica.find_bads_ref(raw_s,method="separate",
                                                        threshold=threshold)
                    else:
                        inds,scores = ica.find_bads_ref(raw,threshold=thresh)
                    #inds = list(filter(lambda idx: idx<ica_cutoff, inds))
                    if len(raw) > signal.shape[1]:
                        raw.crop(tmax=signal.shape[1])
                    else:
                        signal = signal[:,:len(raw)]
                    sraw = mne.io.RawArray(signal,mne.create_info(len(signal),200,ch_types="misc"))
                    for ch_idx,ch in enumerate(sraw.ch_names):
                        sraw.rename_channels({ch:"SRC"+str(ch_idx)})
                    raw.add_channels([sraw],force_update_info=True)
                    gnd_inds,gnd_scores = ica.find_bads_ref(raw,ch_name=sraw.ch_names,
                                                            method="separate",
                                                            threshold=gnd_thresh)
                    #gnd_inds = list(filter(lambda gnd_idx: gnd_idx<ica_cutoff, gnd_inds))
                    temp_hits = list(set(inds) & set(gnd_inds))
                    temp_misses = list(set(gnd_inds) - set(inds))
                    temp_false_alarms = list(set(inds) - set(gnd_inds))
                    for scores_idx,scores in enumerate(gnd_scores):
                        if find_outliers(scores,gnd_thresh).size == 0:
                            silents["rr"].append(constellation["pos"]["rr"][scores_idx])
                            silents["nn"].append(constellation["pos"]["nn"][scores_idx])
                            silents["src_inds"].append(constellation["src_inds"][scores_idx])
                            silents["subj_run"].append((sub,run,n_idx))
                            silents["cor"].append(np.abs(gnd_scores[scores_idx]).max())
                    for comp in temp_hits:
                        gnd_idx = np.argmax([abs(x[comp]) for x in gnd_scores])
                        hits["rr"].append(constellation["pos"]["rr"][gnd_idx])
                        hits["nn"].append(constellation["pos"]["nn"][gnd_idx])
                        hits["src_inds"].append(constellation["src_inds"][gnd_idx])
                        hits["subj_run"].append((sub,run,n_idx))
                        hits["comp"].append((comp,len(ica.ch_names)))
                        hits["cor"].append(gnd_scores[gnd_idx][comp])
                    for comp in temp_misses:
                        gnd_idx = np.argmax([abs(x[comp]) for x in gnd_scores])
                        misses["rr"].append(constellation["pos"]["rr"][gnd_idx])
                        misses["nn"].append(constellation["pos"]["nn"][gnd_idx])
                        misses["src_inds"].append(constellation["src_inds"][gnd_idx])
                        misses["subj_run"].append((sub,run,n_idx))
                        misses["comp"].append((comp,len(ica.ch_names)))
                        misses["cor"].append(gnd_scores[gnd_idx][comp])
                    for comp in temp_false_alarms:
                        gnd_idx = np.argmax([abs(x[comp]) for x in gnd_scores])
                        false_alarms["rr"].append(constellation["pos"]["rr"][gnd_idx])
                        false_alarms["nn"].append(constellation["pos"]["nn"][gnd_idx])
                        false_alarms["src_inds"].append(constellation["src_inds"][gnd_idx])
                        false_alarms["subj_run"].append((sub,run,n_idx))
                        false_alarms["comp"].append((comp,len(ica.ch_names)))
                        false_alarms["cor"].append(gnd_scores[gnd_idx][comp])
                    print("{} hits\n{} misses\n{} false alarms\n{} silents".format(
                    len(hits["rr"]),len(misses["rr"]),len(false_alarms["rr"]),len(silents["rr"])))

        if separate:
            with open(proc_dir+"perform_itr_sep_{}_gndz{}_ica{}".format(thresh,gnd_thresh,cn),"wb") as f:
                pickle.dump({"hits":hits,"misses":misses,"false_alarms":false_alarms,"silents":silents},f)
        else:
            with open(proc_dir+"perform_itr_{}_gndz{}_ica{}".format(thresh,gnd_thresh,cn),"wb") as f:
                pickle.dump({"hits":hits,"misses":misses,"false_alarms":false_alarms,"silents":silents},f)

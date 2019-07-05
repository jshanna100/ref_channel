import mne
import pickle
import numpy as np
from compensate import compensate
import time

proc_dir = "/home/jeff/reftest/proc/"
subjs = ["ATT_10","ATT_11","ATT_12","ATT_13","ATT_14"]
runs = ["2","3","4","5"]
# subjs = ["ATT_11"]
# runs = ["3"]
n_num = (0,50)
threshes = [1,1.5,2,2.5,3,3.5,4]
threshes = [0.9]
gnd_thresh = [0.9]
z_threshes = [2.5]
separate = True
ica_cutoff = 100

for z_thresh,thresh in zip(z_threshes,threshes):
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
                compensate(raw)
                ica = mne.preprocessing.read_ica("{dir}nc_{sub}_{run}_{n}_sim-ica.fif".format(dir=proc_dir,sub=sub,run=run,n=n_idx))
                if separate:
                    ref_ica = mne.preprocessing.read_ica("{dir}nc_{sub}_{run}_{n}_sim_ref-ica.fif".format(dir=proc_dir,sub=sub,run=run,n=n_idx))
                    ref_src = ref_ica.get_sources(raw)
                    for ch_idx,ch in enumerate(ref_src.ch_names):
                        ref_src.rename_channels({ch:"REF_ICA"+str(ch_idx)})
                    raw.add_channels([ref_src])
                    inds,scores = ica.find_bads_ref(raw,method="separate",
                                                    bad_measure="cor",
                                                    threshold=threshold)
                else:
                    inds,scores = ica.find_bads_ref(raw,threshold=z_thresh)
                inds = list(filter(lambda idx: idx<ica_cutoff, inds))
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
                                                        bad_measure="cor",
                                                        threshold=0.3)
                gnd_inds = list(filter(lambda gnd_idx: gnd_idx<ica_cutoff, gnd_inds))
                temp_hits = list(set(inds) & set(gnd_inds))
                temp_misses = list(set(gnd_inds) - set(inds))
                temp_false_alarms = list(set(inds) - set(gnd_inds))
                temp_silent = [gnd_idx if not np.where(gnd>thresh)[0].size else None for gnd_idx,gnd in enumerate(gnd_scores[:ica_cutoff])]
                for sil in temp_silent:
                    if sil is None:
                        continue
                    silents["rr"].append(constellation["pos"]["rr"][sil])
                    silents["nn"].append(constellation["pos"]["nn"][sil])
                    silents["src_inds"].append(constellation["src_inds"][sil])
                    silents["subj_run"].append((sub,run,n_idx))
                    silents["cor"].append(gnd_scores[sil].max())
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
                len(hits["rr"]),len(misses["rr"]),len(false_alarms["rr"]),len(silents["rr"])
                ))
                print("sleep")
                #time.sleep(3)
    if separate:
        with open(proc_dir+"perform_sep_{}_gnd{}_{}".format(thresh,gnd_thresh[0],ica_cutoff),"wb") as f:
            pickle.dump({"hits":hits,"misses":misses,"false_alarms":false_alarms,"silents":silents},f)
    else:
        with open(proc_dir+"perform_{}_{}_{}".format(thresh,z_thresh,ica_cutoff),"wb") as f:
            pickle.dump({"hits":hits,"misses":misses,"false_alarms":false_alarms,"silents":silents},f)

import mne
import pickle
import numpy as np
from compensate import compensate

proc_dir = "/home/jeff/reftest/proc/"
subjs = ["ATT_10","ATT_11","ATT_12","ATT_13","ATT_14"]
runs = ["2","3","4","5"]
# subjs = ["ATT_11"]
# runs = ["3"]
n_num = (0,50)
threshes = [1,1.5,2,2.5,3,3.5,4]

for thresh in threshes:
    threshold = thresh
    hits = {"rr":[],"nn":[],"src_inds":[],"subj_run":[],"comp":[],"cor":[]}
    misses = {"rr":[],"nn":[],"src_inds":[],"subj_run":[],"comp":[],"cor":[]}
    false_alarms = {"rr":[],"nn":[],"src_inds":[],"subj_run":[],"comp":[],"cor":[]}
    for sub in subjs:
        for run in runs:
            for n_idx in range(n_num[0],n_num[1]):
                raw = mne.io.Raw("{dir}nc_{sub}_{run}_{n}_sim-raw.fif".format(dir=proc_dir,sub=sub,run=run,n=n_idx),preload=True)
                compensate(raw)
                ica = mne.preprocessing.read_ica("{dir}nc_{sub}_{run}_{n}_sim-ica.fif".format(dir=proc_dir,sub=sub,run=run,n=n_idx))
                with open("{dir}const_{n}".format(dir=proc_dir,n=n_idx),'rb') as f:
                    constellation = pickle.load(f)
                inds,scores = ica.find_bads_ref(raw,threshold=threshold)
                x2, y2, z2 = constellation["pos"]["rr"].T
                signal = constellation["signal"]
                if len(raw) > signal.shape[1]:
                    raw.crop(tmax=signal.shape[1])
                else:
                    signal = signal[:,:len(raw)]
                sraw = mne.io.RawArray(signal,mne.create_info(len(signal),200,ch_types="misc"))
                signal, ica_arr = sraw.get_data(), ica.get_sources(raw).get_data()
                gesamt = np.concatenate([signal,ica_arr])
                cor = np.corrcoef(gesamt)[:len(signal),len(signal):]
                wh_inds = np.where(np.abs(cor)>0.5)
                for src,comp in zip(list(wh_inds[0]),wh_inds[1]):
                    if comp in inds:
                        hits["rr"].append(constellation["pos"]["rr"][src])
                        hits["nn"].append(constellation["pos"]["nn"][src])
                        hits["src_inds"].append(constellation["src_inds"][src])
                        hits["subj_run"].append((sub,run,n_idx))
                        hits["comp"].append((comp,len(ica_arr)))
                        hits["cor"].append(cor[src,comp])
                    else:
                        misses["rr"].append(constellation["pos"]["rr"][src])
                        misses["nn"].append(constellation["pos"]["nn"][src])
                        misses["src_inds"].append(constellation["src_inds"][src])
                        misses["subj_run"].append((sub,run,n_idx))
                        misses["comp"].append((comp,len(ica_arr)))
                        misses["cor"].append(cor[src,comp])
                for idx in inds:
                    if len(np.where(np.abs(cor[:,idx])>0.5)[0])==0:
                        for src_idx in range(len(signal)):
                            false_alarms["rr"].append(constellation["pos"]["rr"][src_idx])
                            false_alarms["nn"].append(constellation["pos"]["nn"][src_idx])
                            false_alarms["src_inds"].append(constellation["src_inds"][src_idx])
                            false_alarms["subj_run"].append((sub,run,n_idx))
                            false_alarms["comp"].append((idx,len(ica_arr)))
                            false_alarms["cor"].append(cor[src_idx,idx])

    with open(proc_dir+"perform_{}".format(thresh),"wb") as f:
        pickle.dump({"hits":hits,"misses":misses,"false_alarms":false_alarms},f)

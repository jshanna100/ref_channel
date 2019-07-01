import mne
import pickle
import numpy as np
from compensate import compensate
import matplotlib.pyplot as plt
from mayavi import mlab
from scipy.stats import zscore

plt.ion()
proc_dir = "/home/jeff/reftest/proc/"
subjs = ["ATT_10","ATT_11","ATT_12","ATT_13","ATT_14"]
runs = ["2","3","4","5"]
subjs = ["ATT_11"]
runs = ["3"]

n = 3
n_num = (0,20)

threshold = 2.5
filelist = []
for sub in subjs:
    p_idx = 1
    for run in runs:
        plt.figure()
        for n_idx in range(n_num[0],n_num[1]):
            raw = mne.io.Raw("{dir}nc_{sub}_{run}_{n}_sim-raw.fif".format(dir=proc_dir,sub=sub,run=run,n=n_idx),preload=True)
            compensate(raw)
            ica = mne.preprocessing.read_ica("{dir}nc_{sub}_{run}_{n}_sim-ica.fif".format(dir=proc_dir,sub=sub,run=run,n=n_idx))
            with open("{dir}const_{n}".format(dir=proc_dir,n=n_idx),'rb') as f:
                constellation = pickle.load(f)
            inds,scores = ica.find_bads_ref(raw,threshold=threshold)
            x2, y2, z2 = constellation["pos"]["rr"].T

            ml = mlab.figure(1, bgcolor=(0, 0, 0))
            mlab.points3d(x2, y2, z2, color=(1, 1, 0), scale_factor=0.3)
            for p in range(len(x2)):
                mlab.text3d(x2[p], y2[p], z2[p], str(n_idx)+" "+str(p), color=(1, 1, 1))
            mlab.points3d(0, 0, 0, color=(0, 0, 1), scale_factor=0.5)

            signal = constellation["signal"]
            if len(raw) > signal.shape[1]:
                raw.crop(tmax=signal.shape[1])
            else:
                signal = signal[:,:len(raw)]
            sraw = mne.io.RawArray(signal,mne.create_info(len(signal),200,ch_types="misc"))
            signal, ica_arr = sraw.get_data(), ica.get_sources(raw).get_data()
            gesamt = np.concatenate([signal,ica_arr])
            cor = np.corrcoef(gesamt)[:len(signal),len(signal):]
            #cor = zscore(cor,axis=1)
            hit_inds = np.ones((1,len(ica_arr)))*-1
            hit_inds[0,inds] = 1
            cor = np.concatenate([hit_inds,cor])
            ax = plt.subplot(n_num[1]-n_num[0],1,p_idx)
            ax.set_title(str(n_idx))
            plt.imshow(cor,vmin=-1,vmax=1)
            p_idx += 1

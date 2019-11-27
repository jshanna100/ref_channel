import mne
import numpy as np
from numpy.fft import rfft, irfft, rfftfreq
import pickle
import code
import matplotlib.pyplot as plt
plt.ion()

def msr_func(sig_len,freq_ranges,y_targs,sfreq=.005):
    freqs = rfftfreq(sig_len,d=sfreq)
    filtfunc = np.empty(len(freqs))
    for fr,yt in zip(freq_ranges,y_targs):
        freq_inds = (np.searchsorted(freqs,fr[0]),
                     np.searchsorted(freqs,fr[1])+1)
        logvals = np.log10(yt)
        filtfunc[freq_inds[0]:freq_inds[1]] = np.logspace(logvals[0],logvals[1],
                                              num=freq_inds[1]-freq_inds[0])
        filtfunc = 1/filtfunc
        filtfunc = filtfunc + 1j * filtfunc
    return filtfunc

def fft_filter(signal,filter):
    f = rfft(signal)
    f *= filter
    filt_signal = irfft(f)
    return filt_signal.real

proc_dir = "/media/hdd/jeff/reftest/proc/"
# # harvest the ICA "sources"
# sources = []
# maxlen = 0
# subjs = ["ATT_10","ATT_11","ATT_12","ATT_13","ATT_14"]
# runs = ["2","3","4","5"]
# for sub in subjs:
#     for run in runs:
#         raw = mne.io.Raw("{dir}{sub}_{run}-raw.fif".format(dir=proc_dir,sub=sub,run=run))
#         ica = mne.preprocessing.read_ica("{dir}{sub}_{run}_ref-ica.fif".format(dir=proc_dir,sub=sub,run=run))
#         srcs = ica.get_sources(raw).get_data()
#         for s in range(len(srcs)):
#             sources.append(srcs[s,])
#             maxlen = len(srcs[s,]) if len(srcs[s,]) > maxlen else maxlen
# # put them on equal footing with regard to time
# source_array = np.zeros((len(sources),maxlen))
# for s_idx,s in enumerate(sources):
#     randstart = np.random.randint(maxlen-len(s)+1)
#     source_array[s_idx,randstart:randstart+len(s)] = s
# np.save("source_array.npy".format(dir=proc_dir),source_array)

source_array = np.load("source_array.npy".format(dir=proc_dir))

out_src_min = 2
out_src_max = 4
in_src_min = 0
in_src_max = 2
const_size = 100
out_field_constant = 1e-7
in_field_constant = 1e-7
msr_filt = msr_func(source_array.shape[1],[[0,100]],[[100,100000]])

for n in range(const_size):
    if in_src_max > in_src_min:
        in_src_num = np.random.randint(in_src_min,high=in_src_max)
    else:
        in_src_num = 0
    out_src_num = np.random.randint(out_src_min,high=out_src_max)
    src_num = in_src_num + out_src_num
    src_inds = np.random.randint(len(source_array),size=src_num)
    pos = {}
    rr = np.zeros((src_num,3))
    out_data = source_array[src_inds,].copy()
    while np.min(np.linalg.norm(rr,axis=1))<0.5:
        in_src_vec = np.random.randint(1,high=3,size=in_src_num)
        out_src_vec = np.random.randint(5,high=500,size=out_src_num)
        scale_by = np.diag(np.hstack((in_src_vec,out_src_vec)))
        rr = np.dot(scale_by,np.random.random_sample(size=(src_num,3)))
    #code.interact(local=locals())
    for src_idx in range(src_num):
        this_norm = np.linalg.norm(rr[src_idx,])
        rand_coef = np.abs(np.random.normal(1,0.1))
        if this_norm > 3:
            out_data[src_idx,] *= (out_field_constant*rand_coef*this_norm**3)
            orig_ssq = np.sum(out_data[src_idx,]**2)
            out_data[src_idx,] = fft_filter(out_data[src_idx,],msr_filt)
            after_ssq = np.sum(out_data[src_idx,]**2)
            filtmag_coef = orig_ssq/after_ssq
            out_data[src_idx,] *= filtmag_coef
        else:
            out_data[src_idx,] *= (in_field_constant*rand_coef*this_norm**3)
            print("Source inside the MSR")
        # add a bit of noise
        out_data[src_idx,] += np.random.normal(0,out_data[src_idx,].std()/100,
                                               size=out_data[src_idx,].shape)
    #code.interact(local=locals())
    pos["rr"] = rr
    pos["nn"] = np.random.normal(0,1,rr.shape)
    constellation = {"pos":pos,"signal":out_data,"src_inds":src_inds}
    with open("{dir}const_{n}".format(dir=proc_dir,n=n),"wb") as f:
        pickle.dump(constellation,f)

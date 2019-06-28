import mne
import numpy as np
import pickle

def outer_source_gen(src_num,dat_len,sfreq):
    out_data = np.empty((src_num,dat_len))
    switcher = np.zeros((src_num,dat_len))
    for src in range(src_num):
        ball = np.random.random()
        if ball < 0.3333333:
            switcher[src,:] = 1
        elif ball > 0.6666666:
            start = np.random.randint(dat_len)
            finish = np.random.randint(start+1,high=dat_len)
            switcher[src,start:finish] = 1
        else:
            start = np.random.randint(dat_len-2)
            finish = np.random.randint(start+1,high=dat_len)
            switcher[src,start:finish] = 1
            taper_in_finish = np.random.randint(start+1,high=finish-2)
            taper_out_start = np.random.randint(taper_in_finish+1,high=finish)
            switcher[src,start:taper_in_finish] = np.linspace(0,1,num=taper_in_finish-start)
            switcher[src,taper_out_start:finish] = np.linspace(1,0,num=finish-taper_out_start)
        freq = np.random.random()*100+0.1
        phase = np.random.random()*2*np.pi
        amp = np.random.normal(3e-3,scale=1e-3)
        noise_amp = np.random.random()*5e-5+1e-5
        out_data[src,] = np.sin(np.arange(dat_len)/sfreq*freq+phase)*amp
        out_data[src,] += np.random.normal(0,scale=noise_amp,size=out_data[src,].shape)
    out_data *= switcher
    return out_data

proc_dir = "/media/hdd/jeff/reftest/proc/"
# harvest the ICA "sources"
sources = []
maxlen = 0
subjs = ["ATT_10","ATT_11","ATT_12","ATT_13","ATT_14"]
runs = ["2","3","4","5"]
for sub in subjs:
    for run in runs:
        raw = mne.io.Raw("{dir}{sub}_{run}-raw.fif".format(dir=proc_dir,sub=sub,run=run))
        ica = mne.preprocessing.read_ica("{dir}{sub}_{run}_ref-ica.fif".format(dir=proc_dir,sub=sub,run=run))
        srcs = ica.get_sources(raw).get_data()
        for s in range(len(srcs)):
            sources.append(srcs[s,])
            maxlen = len(srcs[s,]) if len(srcs[s,]) > maxlen else maxlen
# put them on equal footing with regard to time
source_array = np.zeros((len(sources),maxlen))
for s_idx,s in enumerate(sources):
    randstart = np.random.randint(maxlen-len(s)+1)
    source_array[s_idx,randstart:randstart+len(s)] = s
source_array *= 2e-3
np.save("source_array.npy".format(dir=proc_dir),source_array)

src_std = 12
src_mean = 0
src_min = 2
src_max = 6
const_size = 50

for n in range(const_size):
    src_num = np.random.randint(src_min,high=src_max)
    src_inds = np.random.randint(len(source_array),size=src_num)
    pos = {}
    rr = np.zeros((src_num,3))
    while np.min(np.linalg.norm(rr,axis=1))<3:
        rr = np.random.normal(src_mean,src_std,(src_num,3))
    pos["rr"] = rr
    pos["nn"] = np.random.normal(0,1,rr.shape)
    #out_data = outer_source_gen(src_num,40000,200)
    out_data = source_array[src_inds,]
    constellation = {"pos":pos,"signal":out_data,"src_inds":src_inds}
    with open("{dir}const_{n}".format(dir=proc_dir,n=n),"wb") as f:
        pickle.dump(constellation,f)

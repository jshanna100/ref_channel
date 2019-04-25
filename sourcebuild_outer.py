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
        amp = np.random.normal(5e-4,scale=1e-4)
        noise_amp = np.random.random()*1e-4+5e-5
        out_data[src,] = np.sin(np.arange(dat_len)/sfreq*freq+phase)*amp
        out_data[src,] += np.random.normal(0,scale=noise_amp,size=out_data[src,].shape)
    out_data *= switcher
    return out_data

proc_dir = "/media/hdd/jeff/reftest/proc/"

src_std = 4
src_mean = 0
src_min = 3
src_max = 20

for n in range(40):
    src_num = np.random.randint(src_min,high=src_max)
    pos = {}
    rr = np.zeros((src_num,3))
    while np.min(np.linalg.norm(rr,axis=1))<1:
        rr = np.random.normal(src_mean,src_std,(src_num,3))
    pos["rr"] = rr
    pos["nn"] = np.random.normal(0,1,rr.shape)
    out_data = outer_source_gen(src_num,85000,200)
    constellation = {"pos":pos,"signal":out_data}
    with open("{dir}const_{n}".format(dir=proc_dir,n=n),"wb") as f:
        pickle.dump(constellation,f)

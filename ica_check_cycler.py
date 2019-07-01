import mne
import matplotlib.pyplot as plt
import numpy as np
import pickle
from compensate import compensate
from mayavi import mlab

# reduces the pain of finding and removing ICA components. Analogous to
# annot_cycler.py

plt.ion()

base_dir ="/media/hdd/jeff/reftest/"
proc_dir = base_dir+"proc/"
subjs = ["ATT_10","ATT_11","ATT_12","ATT_13","ATT_14"]
subjs = ["ATT_14"]
runs = ["2","3","4","5"]
runs=["2"]

filelist = []
for sub in subjs:
    for run in runs:
        for n_idx in range(50):
            filelist.append(["{dir}nc_{sub}_{run}_{n}_sim-raw.fif".format(dir=proc_dir,sub=sub,run=run,n=n_idx),
            "{dir}nc_{sub}_{run}_{n}_sim-ica.fif".format(dir=proc_dir,sub=sub,run=run,n=n_idx),
            "{dir}const_{n}".format(dir=proc_dir,n=n_idx)])

ref_comp_num = 20
class Cycler():

    def __init__(self,filelist,ref_comp_num):
        self.filelist = filelist
        self.ref_comp_num = ref_comp_num

    def go(self,idx=0):
        plt.close('all')
        mlab.close(all=True)
        # load the next raw/ICA files
        self.fn = self.filelist.pop(idx)
        self.raw = mne.io.Raw(self.fn[0],preload=True)
        compensate(self.raw)
        self.ica = mne.preprocessing.read_ica(self.fn[1])

        self.comps = []
        # plot everything out for overview
        self.ica.plot_components(picks=list(range(20)))
        self.ica.plot_sources(self.raw)

        with open(self.fn[2],'rb') as f:
            constellation = pickle.load(f)
        x2, y2, z2 = constellation["pos"]["rr"].T
        signal = constellation["signal"]

        ml = mlab.figure(1, bgcolor=(0, 0, 0))
        mlab.points3d(x2, y2, z2, color=(1, 1, 0), scale_factor=0.3)
        for p in range(len(x2)):
            mlab.text3d(x2[p], y2[p], z2[p], str(p), color=(1, 1, 1))
        mlab.points3d(0, 0, 0, color=(0, 0, 1), scale_factor=0.5)

        self.sraw = mne.io.RawArray(signal,mne.create_info(len(signal),200,ch_types="misc"))
        self.sraw.crop(tmax=self.raw.times[-1])
        self.sraw.plot(scalings="auto",n_channels=len(signal))

    def plot_props(self,props=None):
        # in case you want to take a closer look at a component
        if not props:
            props = self.comps
        self.ica.plot_properties(self.raw,props)

    def show_file(self):
        print("Current raw file: "+self.fn[0])

    def without(self,comps=None,fmax=40):
        # see what the data would look like if we took comps out
        self.comps += self.ica.exclude
        if not comps:
            comps = self.comps
        test = self.raw.copy()
        test.load_data()
        test = self.ica.apply(test,exclude=comps)
        test.plot_psd(fmax=fmax)
        test.plot(duration=30,n_channels=30)
        self.test = test

    def identify_bad(self,method,threshold=3):
        # search for components which correlate with noise
        if isinstance(method,str):
            method = [method]
        elif not isinstance(method,list):
            raise ValueError('"method" must be string or list.')
        for meth in method:
            print(meth)
            if meth == "eog":
                func = self.ica.find_bads_eog
            elif meth == "ecg":
                func = self.ica.find_bads_ecg
            elif meth == "ref":
                func = self.ica.find_bads_ref
            else:
                raise ValueError("Unrecognised method.")
            inds, scores = func(self.raw,threshold=threshold)
            print(inds)
            if inds:
                self.ica.plot_scores(scores, exclude=inds)
                self.comps += inds

    def save(self,comps=None):
        # save the new file
        self.comps += self.ica.exclude
        if not comps:
            self.ica.apply(self.raw,exclude=self.comps).save(self.fn[0][:-8]+"_ica-raw.fif",overwrite=True)
        elif isinstance(comps,list):
            self.ica.apply(self.raw,exclude=self.comps+comps).save(self.fn[0][:-8]+"_ica-raw.fif",overwrite=True)
        else:
            print("No components applied, saving anyway for consistency.")
            self.raw.save(self.fn[0][:-8]+"_ica-raw.fif",overwrite=True)

cyc = Cycler(filelist,ref_comp_num)

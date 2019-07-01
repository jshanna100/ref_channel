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
runs = ["2","3","4","5"]

filelist = []
for sub in subjs:
    for run in runs:
        filelist.append(["{dir}{sub}_{run}-raw.fif".format(dir=proc_dir,sub=sub,run=run),
        "{dir}{sub}_{run}-ica.fif".format(dir=proc_dir,sub=sub,run=run)])

class Cycler():

    def __init__(self,filelist):
        self.filelist = filelist

    def go(self,idx=0):
        plt.close('all')
        # load the next raw/ICA files
        self.fn = self.filelist.pop(idx)
        self.raw = mne.io.Raw(self.fn[0],preload=True)
        #compensate(self.raw)
        self.ica = mne.preprocessing.read_ica(self.fn[1])

        self.comps = []
        # plot everything out for overview
        self.ica.plot_components(picks=list(range(20)))
        self.ica.plot_sources(self.raw)

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

cyc = Cycler(filelist)

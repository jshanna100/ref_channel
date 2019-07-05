from mne.viz.backends.renderer import _Renderer
from mne.bem import read_bem_surfaces
import pickle
import matplotlib.pyplot as plt
import numpy as np


proc_dir = "/home/jeff/reftest/proc/"
with open(proc_dir+"perform_0.5_2.8","rb") as f:
    scores = pickle.load(f)
toviz = ["hits","misses","false_alarms","silents"]
cols = [(1,0,0),(0,0,1),(0,1,0),(1,0,1)]
#toviz = ["misses"]
#cols = [(0,0,1)]

sensor_quiv = np.array(([1,0,0],[0,1,0],[0,0,1]))


renderer = _Renderer(bgcolor=(0, 0, 0), size=(800, 800))
surf = read_bem_surfaces("/home/jeff/mne-python/mne/data/helmets/Magnes_3600wh.fif.gz")
renderer.surface(surface=surf[0])
renderer.quiver3d(0,0,0,sensor_quiv[0,0],sensor_quiv[0,1],sensor_quiv[0,2],
                  color=(1,1,0), mode="arrow", scale=2, opacity=1)
renderer.quiver3d(0,0,0,sensor_quiv[1,0],sensor_quiv[1,1],sensor_quiv[1,2],
                  color=(1,1,0), mode="arrow", scale=2, opacity=1)
renderer.quiver3d(0,0,0,sensor_quiv[2,0],sensor_quiv[2,1],sensor_quiv[2,2],
                  color=(1,1,0), mode="arrow", scale=2, opacity=1)
for tv,col in zip(toviz,cols):
    for sr,coords,direct in zip(scores[tv]["subj_run"],
                                scores[tv]["rr"],scores[tv]["nn"]):
        #if sr[-1] != 33:
        #    continue
        renderer.quiver3d(coords[0], coords[1], coords[2],
                         direct[0], direct[1], direct[2],
                         color=col, mode="arrow", scale=1.5, opacity=0.05)

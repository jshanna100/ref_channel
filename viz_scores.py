from mne.viz.backends.renderer import _Renderer
from mne.bem import read_bem_surfaces
import pickle
import matplotlib.pyplot as plt


proc_dir = "/home/jeff/reftest/proc/"
with open(proc_dir+"perform_2.5","rb") as f:
    scores = pickle.load(f)
toviz = ["hits","misses","false_alarms"]
#toviz = ["misses"]
cols = [(1,0,0),(0,0,1),(0,1,0)]

renderer = _Renderer(bgcolor=(0, 0, 0), size=(800, 800))
surf = read_bem_surfaces("/home/jeff/mne-python/mne/data/helmets/Magnes_3600wh.fif.gz")
renderer.surface(surface=surf[0])
for tv,col in zip(toviz,cols):
    for coords,direct in zip(scores[tv]["rr"],scores[tv]["nn"]):
        renderer.quiver3d(coords[0], coords[1], coords[2],
                         direct[0], direct[1], direct[2],
                         color=col, mode="arrow", scale=1.5, opacity=0.05)

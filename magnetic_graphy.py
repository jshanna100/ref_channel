import numpy as np
import mne
from mne.viz.backends.renderer import _Renderer
from mayavi import mlab

subjects_dir = "/home/jeff/freesurfer/subjects"
subject = "reftest"
surface = "white"
rawfile = "/home/jeff/ATT_dat/proc/nc_ATT_10_1_hand_ica-raw.fif"
raw = mne.io.Raw(rawfile)

#renderer = _Renderer(bgcolor=(0, 0, 0), size=(800, 800))
a = mne.viz.plot_alignment(subject=subject,subjects_dir=subjects_dir,
                           surfaces="brain",meg=["sensors","ref"],
                           info=raw.info)
mlab.quiver3d(0,0.25,-0.25,0,-0.25,0.45,color=(1,0,0))
mlab.quiver3d(0,0,0,0,0,1, color=(1,1,0), scale_factor=0.18)


src = mne.setup_source_space(subject, spacing='oct6',
                             subjects_dir=subjects_dir, add_dist=False)
b = mne.viz.plot_alignment(subject=subject,src=src,surfaces=[],
                           subjects_dir=subjects_dir,meg=["sensors","ref"],
                           info=raw.info)
mlab.quiver3d(.2,.1,.3,0.2,-0.3,0.5, figure=b,
                  color=(1,0,0), mode="arrow", scale_factor=0.3, opacity=1)
mlab.quiver3d(-.2,-.3,-.5,0.8,-0.1,0.2, figure=b,
                  color=(1,0,0), mode="arrow", scale_factor=0.3, opacity=1)
mlab.quiver3d(.5,0,.7,-0.4,-0.6,0.2, figure=b,
                  color=(1,0,0), mode="arrow", scale_factor=0.3, opacity=1)

import mne
import numpy as np

subject = "nc_ATT_10"
subjects_dir = "/home/jeff/freesurfer/subjects/"
trans = "/home/jeff/ATT_dat/proc/nc_ATT_10-trans.fif"
raw = mne.io.Raw("/home/jeff/ATT_dat/proc/nc_ATT_10_1_hand_ica-raw.fif")
all_source = mne.read_source_spaces("/home/jeff/reftest/srcs/all_source-src.fif")

model = mne.make_bem_model(subject=subject, ico=4, conductivity=[0.3],
                               subjects_dir=subjects_dir)
bem = mne.make_bem_solution(model)
fwd = mne.make_forward_solution(raw.info, trans=trans, src=all_source,
                                    bem=bem, n_jobs=8, src_filt=False)

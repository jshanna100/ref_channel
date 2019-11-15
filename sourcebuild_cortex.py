import mne
import numpy as np

subjs = ["ATT_10","ATT_11","ATT_12","ATT_13","ATT_14"]
runs = ["2","3","4","5"]

mri_key = {"KIL13":"ATT_10","ALC81":"ATT_11","EAM11":"ATT_19","ENR41":"ATT_18",
           "NAG_83":"ATT_36","PAG48":"ATT_21","SAG13":"ATT_20","HIU14":"ATT_23",
           "KIL72":"ATT_25","FOT12":"ATT_28","KOI12":"ATT_16","BLE94":"ATT_29",
           "KER27":"ATT_30","ATT_27_fsaverage":"ATT_27","DEN59":"ATT_26",
           "WOO07":"ATT_12","DIU11":"ATT_34","BII41":"ATT_31","Mun79":"ATT_35",
           "ATT_37_fsaverage":"ATT_37","EAM67":"ATT_32","ATT_24_fsaverage":"ATT_24",
           "TGH11":"ATT_14","FIN23":"ATT_17","GIZ04":"ATT_13","BAI97":"ATT_22",
           "WAL70":"ATT_33","ATT_15_fsaverage":"ATT_15"}
mri_key_i = {v:k for k,v in mri_key.items()}
subjects_dir = "/home/jeff/freesurfer/subjects/"
proc_dir = "/media/hdd/jeff/reftest/proc/"

for sub in subjs:
    trans = "{dir}{sub}-trans.fif".format(dir=proc_dir,sub=mri_key_i[sub])
    src = mne.setup_source_space(mri_key_i[sub],surface="white", add_dist=True,
                                 subjects_dir=subjects_dir,n_jobs=8,spacing="oct5")
    src.save("{dir}{sub}-src.fif".format(dir=proc_dir,sub=sub),overwrite=True)
    bem_model = mne.make_bem_model(mri_key_i[sub], conductivity=[0.3],
                                   subjects_dir=subjects_dir)
    bem = mne.make_bem_solution(bem_model)
    mne.write_bem_solution("{dir}{sub}-bem.fif".format(dir=proc_dir,sub=sub),
                           bem)
    # src.plot()
    # mne.viz.plot_bem(subject=subject, subjects_dir=subjects_dir,
    #              brain_surfaces='white', src=src, orientation='coronal')
    src = mne.read_source_spaces("{dir}{sub}-src.fif".format(
                                 dir=proc_dir,sub=sub))
    bem = mne.read_bem_solution("{dir}{sub}-bem.fif".format(
                           dir=proc_dir,sub=sub))
    for run in runs:
        raw = mne.io.Raw("{dir}{sub}_{run}_ica-raw.fif".format(
                         dir=proc_dir,sub=sub,run=run))
        fwd = mne.make_forward_solution(raw.info, trans=trans, src=src, bem=bem,
                                        meg=True, mindist=5.0, n_jobs=4)
        mne.write_forward_solution("{dir}{sub}_{run}-fwd.fif".format(
                                   dir=proc_dir,sub=sub,run=run), fwd,
                                   overwrite=True)

import mne
import numpy as np

subjs = ["ATT_10","ATT_11","ATT_12","ATT_13","ATT_14"]
runs = ["1","2","3","4","5"]
# subjs = ["ATT_14"]
# runs = ["3","4","5"]

subjects_dir = "/home/jeff/freesurfer/subjects/"
proc_dir = "/media/hdd/jeff/reftest/proc/"

for sub in subjs:
    trans = "{dir}nc_{sub}-trans.fif".format(dir=proc_dir,sub=sub)
    src = mne.setup_source_space("nc_"+sub,surface="white", add_dist=True,
                                 subjects_dir=subjects_dir,n_jobs=8,spacing="oct5")
    src.save("{dir}nc_{sub}-src.fif".format(dir=proc_dir,sub=sub),overwrite=True)
    bem_model = mne.make_bem_model("nc_"+sub, conductivity=[0.3],
                                   subjects_dir=subjects_dir)
    bem = mne.make_bem_solution(bem_model)
    mne.write_bem_solution("{dir}nc_{sub}-bem.fif".format(dir=proc_dir,sub=sub),
                           bem)
    # src.plot()
    # mne.viz.plot_bem(subject=subject, subjects_dir=subjects_dir,
    #              brain_surfaces='white', src=src, orientation='coronal')
    src = mne.read_source_spaces("{dir}nc_{sub}-src.fif".format(
                                 dir=proc_dir,sub=sub))
    bem = mne.read_bem_solution("{dir}nc_{sub}-bem.fif".format(
                           dir=proc_dir,sub=sub))
    for run in runs:
        raw = mne.io.Raw("{dir}nc_{sub}_{run}_hand_ica-raw.fif".format(
                         dir=proc_dir,sub=sub,run=run))
        fwd = mne.make_forward_solution(raw.info, trans=trans, src=src, bem=bem,
                                        meg=True, mindist=5.0, n_jobs=4)
        mne.write_forward_solution("{dir}nc_{sub}_{run}-fwd.fif".format(
                                   dir=proc_dir,sub=sub,run=run), fwd,
                                   overwrite=True)

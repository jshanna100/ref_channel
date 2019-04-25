import mne
import numpy as np
import pickle

subjs = ["ATT_10","ATT_11","ATT_12","ATT_13","ATT_14"]
runs = ["1","2","3","4","5"]

subjects_dir = "/home/jeff/freesurfer/subjects/"
proc_dir = "/media/hdd/jeff/reftest/proc/"

dists = []
for sub in subjs:
    trans = "{dir}nc_{sub}-trans.fif".format(dir=proc_dir,sub=sub)
    bem = mne.read_bem_solution("{dir}nc_{sub}-bem.fif".format(
                                dir=proc_dir,sub=sub))
    for run in runs:
        raw = mne.io.Raw("{dir}nc_{sub}_{run}_hand_ica-raw.fif".format(
                         dir=proc_dir,sub=sub,run=run),preload=True)
        stc = mne.read_source_estimate("{dir}nc_{sub}_{run}".format(dir=proc_dir,sub=sub,run=run))
        in_source = mne.read_source_spaces("{dir}nc_{sub}-src.fif".format(dir=proc_dir,
                                           sub=sub,run=run))
        for n_idx in range(40):
            with open("{dir}const_{n}".format(dir=proc_dir,n=n_idx),"rb") as f:
                constellation = pickle.load(f)
            pos = constellation["pos"]
            signal = constellation["signal"]
            sphere = (0,0,0,np.max(np.linalg.norm(pos["rr"],axis=1)))
            out_source = mne.setup_volume_source_space(subject="nc_"+sub,
                                                       pos=pos,sphere=sphere,
                                                       subjects_dir=subjects_dir)
            all_source = out_source + in_source
            fwd = mne.make_forward_solution(raw.info,trans,all_source,bem,
                                            src_filt=False,ref_meg=True,
                                            n_jobs=8)

            if stc.shape[1] > signal.shape[1]:
                stc.crop(tmax=signal.shape[1])
            else:
                signal = signal[:,:stc.shape[1]]
            all_data = np.concatenate((signal,stc.data))
            vertices = [np.arange(len(signal))]+stc.vertices
            all_stc = mne.MixedSourceEstimate(all_data,vertices=vertices,tstep=stc._tstep,tmin=0)
            raw_s = mne.simulation.simulate_raw(raw,all_stc,trans=None,src=None,bem=None,forward=fwd,cov=None,n_jobs=8,ref_meg=True)
            raw_s.save("{dir}nc_{sub}_{run}_{n}_sim-raw.fif".format(
                             dir=proc_dir,sub=sub,run=run,n=n_idx),overwrite=True)



# from mayavi import mlab
# lh_surf = in_source[0]
# # extract left cortical surface vertices, triangle faces, and surface normals
# x1, y1, z1 = lh_surf['rr'].T
# faces = lh_surf['use_tris']
# normals = lh_surf['nn']
# # normalize for mayavi
# normals /= np.sum(normals * normals, axis=1)[:, np.newaxis]
#
# x2, y2, z2 = out_source[0]['rr'][out_source[0]['inuse'].astype(bool)].T
#
# # open a 3d figure in mayavi
# mlab.figure(1, bgcolor=(0, 0, 0))
#
# # plot the outer sources
# mlab.points3d(x2, y2, z2, color=(1, 1, 0), scale_factor=0.1)
#
# # plot the left cortical surface
# mesh = mlab.pipeline.triangular_mesh_source(x1, y1, z1, faces)
# mesh.data.point_data.normals = normals
# mlab.pipeline.surface(mesh, color=3 * (0.7,))

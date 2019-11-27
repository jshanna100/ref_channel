import mne
import numpy as np
import pickle
from mne.forward._compute_forward import _magnetic_dipole_field_vec
from mne.forward._make_forward import _create_meg_coils
import code

mri_key = {"KIL13":"ATT_10","ALC81":"ATT_11","EAM11":"ATT_19","ENR41":"ATT_18",
           "NAG_83":"ATT_36","PAG48":"ATT_21","SAG13":"ATT_20","HIU14":"ATT_23",
           "KIL72":"ATT_25","FOT12":"ATT_28","KOI12":"ATT_16","BLE94":"ATT_29",
           "KER27":"ATT_30","ATT_27_fsaverage":"ATT_27","DEN59":"ATT_26",
           "WOO07":"ATT_12","DIU11":"ATT_34","BII41":"ATT_31","Mun79":"ATT_35",
           "ATT_37_fsaverage":"ATT_37","EAM67":"ATT_32","ATT_24_fsaverage":"ATT_24",
           "TGH11":"ATT_14","FIN23":"ATT_17","GIZ04":"ATT_13","BAI97":"ATT_22",
           "WAL70":"ATT_33","ATT_15_fsaverage":"ATT_15"}
mri_key_i = {v:k for k,v in mri_key.items()}

subjs = ["ATT_10","ATT_11","ATT_12","ATT_13","ATT_14"]
runs = ["2","3","4","5"]
subjects_dir = "/home/jeff/freesurfer/subjects/"
proc_dir = "/media/hdd/jeff/reftest/proc/"

cov = mne.read_cov("{dir}empty-cov.fif".format(dir=proc_dir))

dists = []
for sub in subjs:
    trans = "{dir}{sub}-trans.fif".format(dir=proc_dir,sub=mri_key_i[sub])
    bem = mne.read_bem_solution("{dir}{sub}-bem.fif".format(
                                dir=proc_dir,sub=sub))
    for run in runs:
        raw = mne.io.Raw("{dir}{sub}_{run}_ica-raw.fif".format(
                         dir=proc_dir,sub=sub,run=run),preload=True)
        raw.interpolate_bads()
        stc_master = mne.read_source_estimate("{dir}nc_{sub}_{run}".format(dir=proc_dir,sub=sub,run=run))
        in_source = mne.read_source_spaces("{dir}nc_{sub}-src.fif".format(dir=proc_dir,
                                           sub=sub,run=run))
        for n_idx in range(100):
            # brain sources
            stc = stc_master.copy()
            picks = mne.pick_types(raw.info,meg=True,ref_meg=True)
            info = mne.pick_info(raw.info,picks)
            in_fwd = mne.make_forward_solution(info,trans,in_source,bem,
                                               src_filt=False, ref_meg=True,
                                               n_jobs=8)
            in_raw = mne.simulation.simulate_raw(info,stc,trans=None,
                                                 src=None,bem=None,
                                                 forward=in_fwd,n_jobs=8,
                                                 ref_meg=True)
            mne.simulation.add_noise(in_raw,cov)
            # zero out the brain signal in the reference channels
            ref_picks = mne.pick_types(in_raw.info,meg=False,ref_meg=True)
            in_raw._data[ref_picks,] = np.zeros(in_raw._data[ref_picks,].shape)
            # external sources
            with open("{dir}const_{n}".format(dir=proc_dir,n=n_idx),"rb") as f:
                constellation = pickle.load(f)
            rrs, nns = constellation["pos"]["rr"], constellation["pos"]["nn"]
            signal = constellation["signal"]
            coils = _create_meg_coils(info["chs"],"normal")
            out_fwd = _magnetic_dipole_field_vec(rrs,coils).T
            # set orientations
            out_fwd = np.array([np.dot(out_fwd[:, 3 * ii:3 * (ii + 1)], nns[ii])
                        for ii in range(len(rrs))]).T
            if stc.shape[1] > signal.shape[1]:
                stc.crop(tmax=signal.shape[1])
            else:
                signal = signal[:,:stc.shape[1]]
            out_data = np.dot(out_fwd,signal)
            out_raw = mne.io.RawArray(out_data,info)
            # combine
            comb_raw = in_raw.copy()
            comb_raw._data += out_raw._data
            code.interact(local=locals())
            # out_raw.save("{dir}nc_{sub}_{run}_{n}_outsim-raw.fif".format(
            #            dir=proc_dir,sub=sub,run=run,n=n_idx),overwrite=True)
            # comb_raw.save("{dir}nc_{sub}_{run}_{n}_sim-raw.fif".format(
            #            dir=proc_dir,sub=sub,run=run,n=n_idx),overwrite=True)
            # sraw = mne.io.RawArray(signal,mne.create_info(len(signal),200,ch_types="misc"))




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

# np.linalg.norm(constellation["pos"]["rr"],axis=1)

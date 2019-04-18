import mne
import numpy as np
from mayavi import mlab

subjs = ["ATT_10"]
runs = ["1"]
subjects_dir = "/home/jeff/freesurfer/subjects/"
proc_dir = "/home/jeff/reftest/proc/"

cov = mne.read_cov("{dir}empty_ref-cov.fif".format(dir=proc_dir))
src_std = 4
src_mean = 0
src_num = 3

for sub in subjs:
    for run in runs:
        trans = "{dir}nc_{sub}-trans.fif".format(dir=proc_dir,sub=sub)
        raw = mne.io.Raw("{dir}nc_{sub}_{run}_hand_ica-raw.fif".format(
                         dir=proc_dir,sub=sub,run=run),preload=True)
        bem = mne.read_bem_solution("{dir}nc_{sub}-bem.fif".format(
                                    dir=proc_dir,sub=sub,run=run))
        stc = mne.read_source_estimate("{dir}nc_{sub}_{run}".format(dir=proc_dir,sub=sub,run=run))
        in_source = mne.read_source_spaces("{dir}nc_{sub}-src.fif".format(dir=proc_dir,
                                           sub=sub,run=run))
        pos = {}
        rr = np.zeros((src_num,3))
        while np.min(np.linalg.norm(rr,axis=1))<1:
            rr = np.random.normal(src_mean,src_std,(src_num,3))
        pos["rr"] = rr
        pos["nn"] = np.random.normal(0,1,rr.shape)
        sphere = (0,0,0,np.max(np.linalg.norm(rr,axis=1)))
        out_source = mne.setup_volume_source_space(subject="nc_"+sub,pos=pos,sphere=sphere,
                                                   subjects_dir=subjects_dir)
        all_source = out_source + in_source
        fwd = mne.make_forward_solution(raw.info,trans,all_source,bem,src_filt=False,ref_meg=True,n_jobs=8)

        out_data = np.zeros((src_num,len(raw.times)))
        out_data[0,] = np.sin(np.arange(len(raw.times))/100+15)*2e-3
        out_data[1,] = np.sin(np.arange(len(raw.times))/20+50)*2e-3
        out_data[2,] = np.sin(np.arange(len(raw.times))/50)*2e-3
        print((np.linalg.norm(rr[0,])))
        all_data = np.concatenate((out_data,stc.data))
        vertices = [np.arange(src_num)]+stc.vertices
        all_stc = mne.MixedSourceEstimate(all_data,vertices=vertices,tstep=stc._tstep,tmin=0)
        del stc
        raw_s = mne.simulation.simulate_raw(raw,all_stc,trans=None,src=None,bem=None,forward=fwd,cov=None,n_jobs=8,ref_meg=True)
        raw_s.plot()
        raw_s.save("{dir}test-raw.fif".format(dir=proc_dir),overwrite=True)






#plot
lh_surf = in_source[0]
# extract left cortical surface vertices, triangle faces, and surface normals
x1, y1, z1 = lh_surf['rr'].T
faces = lh_surf['use_tris']
normals = lh_surf['nn']
# normalize for mayavi
normals /= np.sum(normals * normals, axis=1)[:, np.newaxis]

x2, y2, z2 = out_source[0]['rr'][out_source[0]['inuse'].astype(bool)].T

# open a 3d figure in mayavi
mlab.figure(1, bgcolor=(0, 0, 0))

# plot the outer sources
mlab.points3d(x2, y2, z2, color=(1, 1, 0), scale_factor=0.1)

# plot the left cortical surface
mesh = mlab.pipeline.triangular_mesh_source(x1, y1, z1, faces)
mesh.data.point_data.normals = normals
mlab.pipeline.surface(mesh, color=3 * (0.7,))

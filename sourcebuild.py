import mne
import numpy as np
from mayavi import mlab

def outer_source_gen(dat_len,src_num):
    out_data = np.empty((dat_len,src_num))
    switcher = np.zeros((dat_len,src_num))
    for src in range(src_num):
        ball = np.random.random()
        if ball < 0.3333333:
            switcher[:,src] = 1
        elif ball > 0.6666666
            start = np.randint(dat_len)
            finish = np.randint(start+1,high=dat_len)
            switcher[start:finish,src] = 1
        else:
            start = np.randint(dat_len)
            finish = np.randint(start+1,high=dat_len)
            taper_in_start = np.randint(start+1,high=np.int((finish-start)/2))
            taper_out_start = np.randint(taper_in_start+1,high=finish)






subjs = ["ATT_10","ATT_11","ATT_12","ATT_13","ATT_14"]
runs = ["1","2","3","4","5"]

subjects_dir = "/home/jeff/freesurfer/subjects/"
proc_dir = "/media/hdd/jeff/reftest/proc/"

#cov = mne.read_cov("{dir}empty_ref-cov.fif".format(dir=proc_dir))
src_std = 4
src_mean = 0
src_num = 7

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
        out_data[0,] = np.sin(np.arange(len(raw.times))*np.random.random()+np.random.randint(50000))*5e-4
        out_data[1,] = np.sin(np.arange(len(raw.times))*np.random.random()+np.random.randint(50000))*5e-4
        out_data[2,] = np.sin(np.arange(len(raw.times))*np.random.random()+np.random.randint(50000))*5e-4
        out_data[3,] = np.sin(np.arange(len(raw.times))*np.random.random()+np.random.randint(50000))*5e-4
        out_data[4,] = np.sin(np.arange(len(raw.times))*np.random.random()+np.random.randint(50000))*5e-4
        out_data[5,] = np.sin(np.arange(len(raw.times))*np.random.random()+np.random.randint(50000))*5e-4
        out_data[6,] = np.sin(np.arange(len(raw.times))*np.random.random()+np.random.randint(50000))*5e-4
        all_data = np.concatenate((out_data,stc.data))
        vertices = [np.arange(src_num)]+stc.vertices
        all_stc = mne.MixedSourceEstimate(all_data,vertices=vertices,tstep=stc._tstep,tmin=0)
        del stc,out_data
        raw_s = mne.simulation.simulate_raw(raw,all_stc,trans=None,src=None,bem=None,forward=fwd,cov=None,n_jobs=8,ref_meg=True)
        #raw_s.plot()
        raw_s.save("{dir}nc_{sub}_{run}_sim-raw.fif".format(
                         dir=proc_dir,sub=sub,run=run),overwrite=True)
        del raw_s



# #plot
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

import mne
import numpy as np
from mayavi import mlab

sub = "ATT_10"
run = "1"
subjects_dir = "/home/jeff/freesurfer/subjects/"
proc_dir = "/home/jeff/reftest/proc/"
trans = "{dir}nc_{sub}-trans.fif".format(dir=proc_dir,sub=sub)

src_std = 10
src_mean = 0
src_num = 12

pos = {}
rr = np.zeros((12,3))
while np.min(np.linalg.norm(rr,axis=1))<1:
    rr = np.random.normal(src_mean,src_std,(src_num,3))
pos["rr"] = rr
pos["nn"] = np.random.normal(0,1,rr.shape)
sphere = (0,0,0,np.max(np.linalg.norm(rr,axis=1)))
out_source = mne.setup_volume_source_space(subject="nc_"+sub,pos=pos,sphere=sphere,
                                           subjects_dir=subjects_dir)
bem = mne.read_bem_solution("{dir}nc_{sub}-bem.fif".format(
                            dir=proc_dir,sub=sub,run=run))

in_source = mne.read_source_spaces("{dir}nc_{sub}-src.fif".format(dir=proc_dir,
                                   sub=sub,run=run))

all_source = out_source + in_source
all_source.save("{dir}nc_{sub}_comb-src.fif".format(dir=proc_dir,sub=sub,run=run),
                overwrite=True)

raw = mne.io.Raw("{dir}nc_{sub}_{run}_hand_ica-raw.fif".format(
                 dir=proc_dir,sub=sub,run=run))

fwd = mne.make_forward_solution(raw.info,trans,all_source,bem,src_filt=False,ref_meg=True)
mne.write_forward_solution("{dir}nc_{sub}_{run}_comb-fwd.fif".format(
                           dir=proc_dir,sub=sub,run=run),fwd,overwrite=True)

# plot
# lh_surf = in_source[0]
# # extract left cortical surface vertices, triangle faces, and surface normals
# x1, y1, z1 = lh_surf['rr'].T
# faces = lh_surf['use_tris']
# normals = lh_surf['nn']
# # normalize for mayavi
# normals /= np.sum(normals * normals, axis=1)[:, np.newaxis]
#
# # extract left cerebellum cortex source positions
# x2, y2, z2 = out_source[0]['rr'][out_source[0]['inuse'].astype(bool)].T
#
# # open a 3d figure in mayavi
# mlab.figure(1, bgcolor=(0, 0, 0))
#
# # plot the left cortical surface
# mesh = mlab.pipeline.triangular_mesh_source(x1, y1, z1, faces)
# mesh.data.point_data.normals = normals
# mlab.pipeline.surface(mesh, color=3 * (0.7,))
#
# # plot the outer sources
# mlab.points3d(x2, y2, z2, color=(1, 1, 0), scale_factor=0.1)

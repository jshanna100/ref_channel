import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pickle
import scipy.stats
plt.ion()

def find_nearest(arr,val):
    idx = np.abs(arr-val).argmin()
    return idx

def entropy_est(x,bins=1000):
    x = x[x!=0] # don't want the quiet beginnings and ends to count
    (counts,edges) = np.histogram(x,bins=bins)
    entropy = scipy.stats.entropy(counts)
    return entropy

subjs = ["ATT_10","ATT_11","ATT_12","ATT_13","ATT_14"]
proc_dir = "/home/jeff/reftest/proc/"
with open(proc_dir+"perform_sep_0.9_gnd0.9_100","rb") as f:
    scores = pickle.load(f)
toviz = ["hits","misses","silents"]
cols = np.array([(1,0,0,0),(0,0,1,0),(0,1,0,0)])
cols = np.array([(1,0,0),(0,0,1),(0,1,0)])

plt.figure()
phit = plt.scatter(0,0,c="red",label="hit",s=50)
pmiss = plt.scatter(0,0,c="blue",label="miss",s=50)
phit.remove()
pmiss.remove()
rc_grid = np.zeros((20,50,3))
for tv_idx,tv in enumerate(["hits","misses","false_alarms"]):
    for rec in scores[tv]["subj_run"]:
        row_idx = subjs.index(rec[0])*4 + int(rec[1])-2
        col_idx = rec[2]
        rc_grid[row_idx,col_idx,] += cols[tv_idx,]*0.03
rc_grid = np.concatenate((rc_grid,np.mean(rc_grid,axis=0,keepdims=True)))
rc_grid = np.concatenate((rc_grid,np.mean(rc_grid,axis=1,keepdims=True)),axis=1)
rc_grid = rc_grid/np.linalg.norm(rc_grid,axis=2,keepdims=True)
plt.xlabel("Noise constellation", fontdict={"size":20})
plt.xticks(ticks=[],labels=[])
plt.ylabel("Brain constellation", fontdict={"size":20})
plt.yticks(ticks=[],labels=[])
plt.imshow(rc_grid)
plt.legend([phit,pmiss],["hit","miss"],prop={"size":20})
plt.suptitle("Accuracy by brain/noise constellations",size=25)

plt.figure()
cs_grid = np.zeros((50,100,3))
for tv_idx,tv in enumerate(["hits","misses","false_alarms"]):
    for rec,src_idx in zip(scores[tv]["subj_run"],scores[tv]["src_inds"]):
        row_idx = rec[2]
        col_idx = src_idx
        cs_grid[row_idx,col_idx,] += cols[tv_idx,]*0.03
cs_grid = np.concatenate((cs_grid,np.mean(cs_grid,axis=0,keepdims=True)))
cs_grid = np.concatenate((cs_grid,np.mean(cs_grid,axis=1,keepdims=True)),axis=1)
cs_grid = cs_grid/np.linalg.norm(cs_grid,axis=2,keepdims=True)
plt.imshow(cs_grid)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
phit = ax.scatter(0,0,0,c="red",label="hit",s=50)
pmiss = ax.scatter(0,0,0,c="blue",label="miss",s=50)
phit.remove()
pmiss.remove()
point_info = {x:[] for x in toviz}
for tv_idx,tv in enumerate(toviz):
    for subj_run,src_idx,rr,nn in zip(scores[tv]["subj_run"],scores[tv]["src_inds"],
                                      scores[tv]["rr"],scores[tv]["nn"]):
        with open("{dir}const_{n}".format(dir=proc_dir,n=subj_run[-1]),'rb') as f:
            constellation = pickle.load(f)
        src = constellation["signal"][np.where(constellation["src_inds"]==src_idx)[0]]
        entropy = entropy_est(src[0],bins=1000)
        dist = np.linalg.norm(rr)
        vec2cent = rr/dist
        nn = nn/np.linalg.norm(nn)
        orth = abs(np.dot(nn,vec2cent))
        point_info[tv].append({"rr":tuple(rr),"dist":dist,"orth":orth,"entropy":entropy})

point_col = {}
for tv in ["hits","misses"]:
    c = np.array((1,0,0,0.05)) if tv == "hits" else np.array((0,0,1,0.05))
    for pi in point_info[tv]:
        if (pi["dist"],pi["orth"],pi["entropy"]) in point_col.keys():
            point_col[(pi["dist"],pi["orth"],pi["entropy"])] = point_col[(pi["dist"],pi["orth"],pi["entropy"])] + c
        else:
            point_col[(pi["dist"],pi["orth"],pi["entropy"])] = c
for k,v in zip(point_col.keys(),point_col.values()):
    normed_col = v[:3]/np.linalg.norm(v[:3])
    col_alpha = (*normed_col,v[-1])
    col = np.expand_dims(col_alpha,0)
    col[0,-1] = 1 if col[0,-1]>1 else col[0,-1]
    ax.scatter(k[0],k[1],k[2],c=col,s=50)
plt.xlabel("Noise source distance to sensors",fontdict={"size":20})
plt.ylabel("Noise source radial to sensors",fontdict={"size":20})
ax.set_zlabel("Noise source estimated entropy",fontdict={"size":20})
plt.legend([phit,pmiss],["hit","miss"],prop={"size":20})
plt.suptitle("Accuracy by distance, orthogonality, and entropy",size=25)

plt.figure()
silent_ent = np.array([x["entropy"] for x in point_info["silents"]])
violent_ent = np.array([x["entropy"] for x in point_info["hits"]]+[x["entropy"] for x in point_info["misses"]])
plt.bar([1,2],[np.mean(silent_ent),np.mean(violent_ent)],yerr=[scipy.stats.sem(silent_ent),scipy.stats.sem(violent_ent)])
plt.ylim((5,6))
plt.suptitle("Estimated entropy of noise sources",size=25)
plt.xticks([1,2],["Removed by traditional method","Not"], fontdict={"size":20})

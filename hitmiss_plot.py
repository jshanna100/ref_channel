import pickle
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

threshes = [.2,.3,.4,.5,.6,.7,.8,.9]
z_threshes = [2.5,3,3.5,4]
threshes = z_threshes
gnd_thresh = 3
ica_cutoff = 20
dir = "/home/jeff/reftest/proc/"
fs = 32

fig, axes = plt.subplots(1,2)

plt.rcParams.update({'font.size': fs})

acc = np.empty((4,len(threshes)))
for thresh_idx,thresh in enumerate(threshes):
    with open("{dir}perform_itr_sep_{thresh}_gndz{gnd_thresh}_ica{ica_cutoff}".format(
              dir=dir,thresh=thresh,gnd_thresh=gnd_thresh,
              ica_cutoff=ica_cutoff), "rb") as f:
        sep_hmcalc = pickle.load(f)
    acc[0,thresh_idx] = len(sep_hmcalc["hits"]["rr"])
    acc[1,thresh_idx] = len(sep_hmcalc["misses"]["rr"])
    acc[2,thresh_idx] = len(sep_hmcalc["false_alarms"]["rr"])
    acc[3,thresh_idx] = len(sep_hmcalc["silents"]["rr"])

dists = np.linalg.norm(np.array(sep_hmcalc["hits"]["rr"]),axis=1)
print("Hits: {},{}".format(np.mean(dists),np.std(dists)))
dists = np.linalg.norm(np.array(sep_hmcalc["misses"]["rr"]),axis=1)
print("Misses: {},{}".format(np.mean(dists),np.std(dists)))
if sep_hmcalc["false_alarms"]["rr"]:
    dists = np.linalg.norm(np.array(sep_hmcalc["false_alarms"]["rr"]),axis=1)
    print("False alarms: {},{}".format(np.mean(dists),np.std(dists)))

axes[0].plot(acc[0,:],label="hits", linewidth=5)
axes[0].plot(acc[1,:],label="misses", linewidth=5)
axes[0].plot(acc[2,:],label="false alarms", linewidth=5)
#plt.plot(acc[3,:],label="silents")
axes[0].set_title("Separate")
#axes[0].ylim((0,2700))
axes[0].legend()
axes[0].set_xticks(np.arange(len(threshes)))
axes[0].set_xticklabels([str(x) for x in threshes], fontsize=fs)
axes[0].set_xlabel("Z threshold", fontsize=fs)
axes[0].set_ylabel("# of identified components", fontsize=fs)
axes[0].yaxis.tick_right()
axes[0].set_yticks(np.arange(0,5000,1000))
axes[0].set_yticklabels(["                          "+str(s) for s in np.arange(0,5000,1000)],
                        fontsize=fs, horizontalalignment="center")
print("Separate:")
print(acc)


acc = np.empty((4,len(z_threshes)))
for thresh_idx,thresh in enumerate(z_threshes):
    with open("{dir}perform_itr_{thresh}_gndz{gnd_thresh}_ica{ica_cutoff}".format(
              dir=dir,thresh=thresh,gnd_thresh=gnd_thresh,
              ica_cutoff=ica_cutoff), "rb") as f:
        hmcalc = pickle.load(f)
    acc[0,thresh_idx] = len(hmcalc["hits"]["rr"])
    acc[1,thresh_idx] = len(hmcalc["misses"]["rr"])
    acc[2,thresh_idx] = len(hmcalc["false_alarms"]["rr"])
    acc[3,thresh_idx] = len(hmcalc["silents"]["rr"])

axes[1].plot(acc[0,:],label="hits", linewidth=5)
axes[1].plot(acc[1,:],label="misses", linewidth=5)
axes[1].plot(acc[2,:],label="false alarms", linewidth=5)
#plt.plot(acc[3,:],label="silents")
axes[1].set_xticks(np.arange(len(threshes)))
axes[1].set_xticklabels([str(x) for x in threshes], fontsize=fs)
axes[1].set_xlabel("Z threshold", fontsize=fs)
axes[1].set_ylim(axes[0].get_ylim())
axes[1].set_title("Together")
axes[1].legend()
axes[1].set_xticks(np.arange(len(threshes)))
axes[1].set_xticklabels([str(x) for x in threshes])
axes[1].set_yticklabels([])
print("Together:")
print(acc)

dists = np.linalg.norm(np.array(hmcalc["hits"]["rr"]),axis=1)
print("Hits: {},{}".format(np.mean(dists),np.std(dists)))
dists = np.linalg.norm(np.array(hmcalc["misses"]["rr"]),axis=1)
print("Misses: {},{}".format(np.mean(dists),np.std(dists)))
if hmcalc["false_alarms"]["rr"]:
    dists = np.linalg.norm(np.array(hmcalc["false_alarms"]["rr"]),axis=1)
    print("False alarms: {},{}".format(np.mean(dists),np.std(dists)))

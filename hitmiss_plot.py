import pickle
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

threshes = [.2,.3,.4,.5,.6,.7,.8,.9]
z_threshes = [2.5,3,3.5,4]
threshes = z_threshes
gnd_thresh = 3
ica_cutoffs = [20,40,60,80,100]
dir = "/home/jeff/reftest/proc/"
fs = 32
distcut = 3000

fig, axes = plt.subplots(2,5)
plt.rcParams.update({'font.size': fs})
for ica_idx,ica_cutoff in enumerate(ica_cutoffs):
    acc = np.empty((3,len(threshes)))
    for thresh_idx,thresh in enumerate(threshes):
        with open("{dir}perform_itr_sep_{thresh}_gndz{gnd_thresh}_ica{ica_cutoff}".format(
                  dir=dir,thresh=thresh,gnd_thresh=gnd_thresh,
                  ica_cutoff=ica_cutoff), "rb") as f:
            sep_hmcalc = pickle.load(f)
        acc[0,thresh_idx] = len(list(filter(lambda x:np.linalg.norm(x)<distcut,sep_hmcalc["hits"]["rr"])))
        acc[1,thresh_idx] = len(list(filter(lambda x:np.linalg.norm(x)<distcut,sep_hmcalc["misses"]["rr"])))
        acc[2,thresh_idx] = len(list(filter(lambda x:np.linalg.norm(x)<distcut,sep_hmcalc["false_alarms"]["rr"])))
        print("Total: {}".format(np.sum(acc[:,thresh_idx])))
        acc[:,thresh_idx] /= np.sum(acc[:,thresh_idx])

    axes[0,ica_idx].plot(acc[0,:],label="hits", linewidth=5)
    axes[0,ica_idx].plot(acc[1,:],label="misses", linewidth=5)
    axes[0,ica_idx].plot(acc[2,:],label="false alarms", linewidth=5)
    if ica_idx == 0:
        axes[0,ica_idx].set_yticks(np.linspace(0.2,0.8,3))
        axes[0,ica_idx].set_yticklabels([str(s) for s in np.linspace(0.2,0.8,3)],
                                fontsize=fs, horizontalalignment="right")
        axes[0,ica_idx].set_ylabel("proportion of total", fontsize=fs)
    else:
        axes[0,ica_idx].set_yticklabels([])
    if ica_idx == 4:
        axes[0,ica_idx].legend()
    if ica_idx == 2:
        axes[0,ica_idx].set_title("{} components\nSeparate".format(ica_cutoff))
    else:
        axes[0,ica_idx].set_title("{} components\n".format(ica_cutoff))
    axes[0,ica_idx].set_xticks(np.arange(len(threshes)))
    axes[0,ica_idx].set_xticklabels([str(x) for x in threshes], fontsize=fs)
    axes[0,ica_idx].set_ylim((0,1))

    print("Separate:")
    print(acc)


    acc = np.empty((3,len(z_threshes)))
    for thresh_idx,thresh in enumerate(z_threshes):
        with open("{dir}perform_itr_{thresh}_gndz{gnd_thresh}_ica{ica_cutoff}".format(
                  dir=dir,thresh=thresh,gnd_thresh=gnd_thresh,
                  ica_cutoff=ica_cutoff), "rb") as f:
            hmcalc = pickle.load(f)
        acc[0,thresh_idx] = len(list(filter(lambda x:np.linalg.norm(x)<distcut,hmcalc["hits"]["rr"])))
        acc[1,thresh_idx] = len(list(filter(lambda x:np.linalg.norm(x)<distcut,hmcalc["misses"]["rr"])))
        acc[2,thresh_idx] = len(list(filter(lambda x:np.linalg.norm(x)<distcut,hmcalc["false_alarms"]["rr"])))
        print("Total: {}".format(np.sum(acc[:,thresh_idx])))
        acc[:,thresh_idx] /= np.sum(acc[:,thresh_idx])

    axes[1,ica_idx].plot(acc[0,:],label="hits", linewidth=5)
    axes[1,ica_idx].plot(acc[1,:],label="misses", linewidth=5)
    axes[1,ica_idx].plot(acc[2,:],label="false alarms", linewidth=5)
    axes[1,ica_idx].set_xticks(np.arange(len(threshes)))
    axes[1,ica_idx].set_xticklabels([str(x) for x in threshes], fontsize=fs)
    if ica_idx == 2:
        axes[1,ica_idx].set_xlabel("Z threshold", fontsize=fs)
        axes[1,ica_idx].set_title("Together")
    if ica_idx == 0:
        axes[1,ica_idx].set_yticks(np.linspace(0.2,0.8,3))
        axes[1,ica_idx].set_yticklabels([str(s) for s in np.linspace(0.2,0.8,3)],
                                fontsize=fs, horizontalalignment="right")
        axes[1,ica_idx].set_ylabel("proportion of total", fontsize=fs)
    else:
        axes[1,ica_idx].set_yticklabels([])
    axes[1,ica_idx].set_xticks(np.arange(len(threshes)))
    axes[1,ica_idx].set_ylim((0,1))
    axes[1,ica_idx].set_xticklabels([str(x) for x in threshes])

    print("Together:")
    print(acc)

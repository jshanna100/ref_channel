import matplotlib.pyplot as plt
plt.ion()
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.model_selection import cross_val_score
import mne
from compensate import compensate
import numpy as np
import pickle

proc_dir = "/media/hdd/jeff/reftest/proc/"
proc_dir = "/home/jeff/reftest/proc/"
subjs = ["ATT_10","ATT_11","ATT_12","ATT_13","ATT_14"]
runs = ["2","3","4","5"]

n_nums = 100
#comp_nums = [25,50,75,100,125,150,175,200,225]
sep_comp_nums = [5,10,15,20,25,30,35,40,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125]
tog_comp_nums = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
tog_comp_nums = sep_comp_nums
n_jobs = 1
sample_size = 200
filenames = []
out_filenames = []
for sub in subjs:
    for run in runs:
        for n_idx in range(n_nums):
            filenames.append("{dir}nc_{sub}_{run}_{n}_sim-raw.fif".format(
                                  dir=proc_dir,
                                  sub=sub,
                                  run=run,n=n_idx))
            out_filenames.append("{dir}nc_{sub}_{run}_{n}_outsim-raw.fif".format(
                                  dir=proc_dir,
                                  sub=sub,
                                  run=run,n=n_idx))
rand_samps = np.random.randint(len(filenames),size=sample_size)
results = {"sep":{"pca_cv":np.zeros((sample_size,len(sep_comp_nums))),"fa_cv":
           np.zeros((sample_size,len(sep_comp_nums))),"pca_screes":
           np.zeros((sample_size,sep_comp_nums[-1])),"pca_ratios":
           np.zeros((sample_size,sep_comp_nums[-1]))},"tog":{"pca_cv":
           np.zeros((sample_size,len(tog_comp_nums))),"fa_cv":
           np.zeros((sample_size,len(tog_comp_nums))),"pca_screes":
           np.zeros((sample_size,tog_comp_nums[-1])),"pca_ratios":
           np.zeros((sample_size,tog_comp_nums[-1]))}}
for i_idx,i in enumerate(np.nditer(rand_samps)):
    raw = mne.io.Raw(filenames[i])
    out_raw = mne.io.Raw(out_filenames[i])
    compensate(raw,template=out_raw)
    picks = mne.pick_types(raw.info,meg=True,ref_meg=True)
    dat_picks = mne.pick_types(raw.info,meg=True,ref_meg=False)
    data = raw.get_data().T
    data *= 1e+13
    pca = PCA(svd_solver='full')
    for comp_idx,(sep_comp,tog_comp) in enumerate(zip(sep_comp_nums,tog_comp_nums)):
        print("{} {}".format(filenames[i],comp_idx))
        pca.n_components = tog_comp

        results["tog"]["pca_cv"][i_idx,comp_idx,] = np.mean(cross_val_score(pca,
                                                            data, cv=5,
                                                            n_jobs=n_jobs))

        pca.n_components = sep_comp
        results["sep"]["pca_cv"][i_idx,comp_idx,] = np.mean(cross_val_score(pca,
                                                            data[:,dat_picks],
                                                            cv=5,
                                                            n_jobs=n_jobs))

        if sep_comp == sep_comp_nums[-1]:
            pca.n_components = tog_comp
            pca.fit(data)
            results["tog"]["pca_screes"][i_idx,] = pca.explained_variance_
            results["tog"]["pca_ratios"][i_idx,] = np.cumsum(
                                                   pca.explained_variance_ratio_)
            pca.n_components = sep_comp
            pca.fit(data[:,dat_picks])
            results["sep"]["pca_screes"][i_idx,] = pca.explained_variance_
            results["sep"]["pca_ratios"][i_idx,] = np.cumsum(
                                                   pca.explained_variance_ratio_)
    with open("{dir}est_results".format(dir=proc_dir),"wb") as f:
        pickle.dump(results,f)

# with open("{dir}est_results".format(dir=proc_dir),"rb") as f:
#     results = pickle.load(f)
#
# sep_pca_cv_fig_ = plt.figure()
# plt.title("Cross validated - separate")
# this_meas = results["sep"]["pca_cv"]
# for ln_idx in range(this_meas.shape[0]):
#     plt.plot(sep_comp_nums,this_meas[ln_idx,],color="blue",alpha=0.05)
# mn = np.mean(this_meas,axis=0)
# plt.plot(sep_comp_nums,mn,color="blue",linewidth=5)
#
# sep_pca_screes_fig_ = plt.figure()
# plt.title("Scree plot - separate")
# this_meas = results["sep"]["pca_screes"]
# for ln_idx in range(this_meas.shape[0]):
#     plt.plot(np.arange(sep_comp_nums[-1]),this_meas[ln_idx,],color="blue",alpha=0.05)
# mn = np.mean(this_meas,axis=0)
# plt.plot(np.arange(sep_comp_nums[-1]),mn,color="blue",linewidth=5)
# plt.axhline(1)
# plt.ylim(0,5)
#
# sep_pca_ratios_fig_ = plt.figure()
# plt.title("Cumulative explained variance - separate")
# this_meas = results["sep"]["pca_ratios"]
# for ln_idx in range(this_meas.shape[0]):
#     plt.plot(np.arange(sep_comp_nums[-1]),this_meas[ln_idx,],color="blue",alpha=0.05)
# mn = np.mean(this_meas,axis=0)
# plt.plot(np.arange(sep_comp_nums[-1]),mn,color="blue",linewidth=5)
#
# tog_pca_cv_fig_ = plt.figure()
# plt.title("Cross validated - together")
# this_meas = results["tog"]["pca_cv"]
# for ln_idx in range(this_meas.shape[0]):
#     plt.plot(tog_comp_nums,this_meas[ln_idx,],color="blue",alpha=0.05)
# mn = np.mean(this_meas,axis=0)
# plt.plot(tog_comp_nums,mn,color="blue",linewidth=5)
#
# tog_pca_screes_fig_ = plt.figure()
# plt.title("Scree plot - together")
# this_meas = results["tog"]["pca_screes"]
# for ln_idx in range(this_meas.shape[0]):
#     plt.plot(np.arange(tog_comp_nums[-1]),this_meas[ln_idx,],color="blue",alpha=0.05)
# mn = np.mean(this_meas,axis=0)
# plt.plot(np.arange(tog_comp_nums[-1]),mn,color="blue",linewidth=5)
# plt.axhline(1)
# plt.ylim(0,5)
#
# tog_pca_ratios_fig_ = plt.figure()
# plt.title("Cumulative explained variance - together")
# this_meas = results["tog"]["pca_ratios"]
# for ln_idx in range(this_meas.shape[0]):
#     plt.plot(np.arange(tog_comp_nums[-1]),this_meas[ln_idx,],color="blue",alpha=0.05)
# mn = np.mean(this_meas,axis=0)
# plt.plot(np.arange(tog_comp_nums[-1]),mn,color="blue",linewidth=5)

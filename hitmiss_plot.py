import pickle
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

threshes = [.2,.3,.4,.5,.6,.7,.8,.9]
z_threshes = [1,1.5,2,2.5,3,3.5,4]
gnd_thresh = 3
ica_cutoff = 300
dir = "/home/jeff/reftest/proc/"

acc = np.empty((4,len(threshes)))
for thresh_idx,thresh in enumerate(threshes):
    with open("{dir}perform_sep_{thresh}_p{gnd_thresh}_{ica_cutoff}".format(
              dir=dir,thresh=thresh,gnd_thresh=gnd_thresh,
              ica_cutoff=ica_cutoff), "rb") as f:
        hmcalc = pickle.load(f)
    acc[0,thresh_idx] = len(hmcalc["hits"]["rr"])
    acc[1,thresh_idx] = len(hmcalc["misses"]["rr"])
    acc[2,thresh_idx] = len(hmcalc["false_alarms"]["rr"])
    acc[3,thresh_idx] = len(hmcalc["silents"]["rr"])

plt.figure()
plt.plot(acc[0,:],label="hits")
plt.plot(acc[1,:],label="misses")
plt.plot(acc[2,:],label="false alarms")
plt.plot(acc[3,:],label="silents")
plt.title("Separate")
plt.ylim((0,1800))
plt.legend()
print("Separate:")
print(acc)


# acc = np.empty((4,len(z_threshes)))
# for thresh_idx,thresh in enumerate(z_threshes):
#     with open("{dir}perform_{thresh}_{gnd_thresh}_{ica_cutoff}".format(
#               dir=dir,thresh=thresh,gnd_thresh=gnd_thresh,
#               ica_cutoff=ica_cutoff), "rb") as f:
#         hmcalc = pickle.load(f)
#     acc[0,thresh_idx] = len(hmcalc["hits"]["rr"])
#     acc[1,thresh_idx] = len(hmcalc["misses"]["rr"])
#     acc[2,thresh_idx] = len(hmcalc["false_alarms"]["rr"])
#     acc[3,thresh_idx] = len(hmcalc["silents"]["rr"])
#
# plt.figure()
# plt.plot(acc[0,:],label="hits")
# plt.plot(acc[1,:],label="misses")
# plt.plot(acc[2,:],label="false alarms")
# #plt.plot(acc[3,:],label="silents")
# plt.ylim((0,1800))
# plt.title("Together")
# plt.legend()
# print("Together:")
# print(acc)

import numpy as np
import pickle

with open("/home/jeff/reftest/bin/Supine1.0.txt","r") as f:
    lines = f.readlines()

analog_weights = {}
digital_weights = {}
for l in lines:
    lsp = l.split()
    if lsp[0] == "Entry":
        analog_comp_names = lsp[2:5]
        digital_comp_names = lsp[5:]
    if lsp[0] != "#" and lsp[1][0] == "A":
        analog_weights[lsp[1]] = lsp[2:5]
        digital_weights[lsp[1]] = lsp[5:]
#TODO fix this to accomodate weights dict with analog and digital entries
# out_str = "{chan_num} {comp_num}".format(chan_num=chan_num,comp_num=comp_num)
# for cn in comp_names:
#     out_str += " " + cn
# for k in weights.keys():
#     out_str += " " + k
#     for w in weights[k]:
#         out_str += " " + w
# with open("/home/jeff/reftest/bin/compsup1.txt","w") as f:
#     f.write(out_str)
analog_comp = {"chan_num":248,"comp_num":3,"comp_names":analog_comp_names,
        "weights":analog_weights}
digital_comp = {"chan_num":248,"comp_num":23,"comp_names":digital_comp_names,
        "weights":digital_weights}
with open("/home/jeff/reftest/bin/compsup1","wb") as f:
    pickle.dump({"analog":analog_comp,"digital":digital_comp},f)

import mne
from compensate import compensate
raw = mne.io.Raw("/home/jeff/ATT_dat/proc/nc_ATT_17_5_hand-raw.fif")
raw_n = raw.copy()
compensate(raw_n,digital_comp,direction=-1)
raw_nn = raw_n.copy()
compensate(raw_nn,direction=1)
raw.plot()
raw_n.plot()
raw_nn.plot()

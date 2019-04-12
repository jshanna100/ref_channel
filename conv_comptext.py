import numpy as np
import pickle

with open("/home/jeff/reftest/bin/Supine1.0.txt","r") as f:
    lines = f.readlines()

weights = {}
chan_num = 248
comp_num = 23
for l in lines:
    lsp = l.split()
    if lsp[0] == "Entry":
        comp_names = lsp[5:]
    if lsp[0] != "#" and lsp[1][0] == "A":
        weights[lsp[1]] = lsp[5:]
out_str = "{chan_num} {comp_num}".format(chan_num=chan_num,comp_num=comp_num)
for cn in comp_names:
    out_str += " " + cn
for k in weights.keys():
    out_str += " " + k
    for w in weights[k]:
        out_str += " " + w
with open("/home/jeff/reftest/bin/compsup1.txt","w") as f:
    f.write(out_str)
comp = {"chan_num":chan_num,"comp_num":comp_num,"comp_names":comp_names,
        "weights":weights}
with open("/home/jeff/reftest/bin/compsup1","wb") as f:
    pickle.dump(comp,f)

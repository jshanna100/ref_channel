import pickle
import matplotlib.pyplot as plt


proc_dir = "/home/jeff/reftest/proc/"
with open(proc_dir+"perform_3","rb") as f:
    scores = pickle.load(f)
toviz = ["hits","misses","false_alarms"]
cols = [(1,0,0),(0,0,1),(0,1,0)]

sources = [{tv:[] for tv in toviz} for x in range(100)]

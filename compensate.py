import mne
import numpy as np

def compensate(raw,weights=None):
    if not raw.preload:
        raw.load_data()
    if not weights:
        meg_picks = mne.pick_types(raw.info,meg=True,ref_meg=False)
        ref_picks = mne.pick_types(raw.info,meg=False,ref_meg=True)
        meg_ch_names = [raw.ch_names[x] for x in meg_picks]
        ref_ch_names = [raw.ch_names[x] for x in ref_picks]
    else:
        meg_picks = mne.pick_types(raw.info,meg=True,ref_meg=False,exclude=[])
        ref_picks = mne.pick_types(raw.info,meg=False,ref_meg=True,exclude=[])
        meg_ch_names = [raw.ch_names[x] for x in meg_picks]
        ref_ch_names = [raw.ch_names[x] for x in ref_picks]
        # build matrix, rows and columns don't correspond at first
        comp_names = weights["comp_names"]
        comp_mat = np.zeros((weights["chan_num"],weights["comp_num"]))
        weights = weights["weights"]
        comp_to_ref = np.array([comp_names.index(x) for x in ref_ch_names])
        for ch_idx,ch in zip(meg_picks,meg_ch_names):
            comp_mat[ch_idx,:] = np.array(weights[ch])[comp_to_ref]
        megdata = np.dot(comp_mat,raw._data[ref_picks,])
        raw._data[meg_picks,] -= megdata

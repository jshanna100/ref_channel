import mne
import numpy as np

def compensate(raw,weights=None,template=None,direction=1):
    if not raw.preload:
        raw.load_data()
    if not weights:
        from sklearn.linear_model import LinearRegression
        meg_picks = mne.pick_types(raw.info,meg=True,ref_meg=False)
        ref_picks = mne.pick_types(raw.info,meg=False,ref_meg=True)
        meg_ch_names = [raw.ch_names[x] for x in meg_picks]
        ref_ch_names = [raw.ch_names[x] for x in ref_picks]
        estimator = LinearRegression(normalize=True)
        if template:
            estimator.fit(template[ref_picks][0].T,template[meg_picks][0].T)
        else:
            estimator.fit(raw[ref_picks][0].T,raw[meg_picks][0].T)
        Y_pred = estimator.predict(raw[ref_picks][0].T)
        raw._data[meg_picks] -= direction*Y_pred.T
        return estimator
    else:
        meg_picks = mne.pick_types(raw.info,meg=True,ref_meg=False,exclude=[])
        ref_picks = mne.pick_channels(raw.ch_names,weights["comp_names"])
        meg_ch_names = [raw.ch_names[x] for x in meg_picks]
        ref_ch_names = [raw.ch_names[x] for x in ref_picks]
        # build matrix, rows and columns don't correspond at first
        comp_names = weights["comp_names"]
        comp_mat = np.zeros((weights["chan_num"],weights["comp_num"]))
        weights = weights["weights"]
        for ch_idx,ch in zip(meg_picks,meg_ch_names):
            comp_mat[ch_idx,:] = np.array(weights[ch])
        megdata = np.dot(comp_mat,raw._data[ref_picks,])
        raw._data[meg_picks,] -= direction*megdata

import mne

base_dir ="/home/jeff/reftest/"
proc_dir = base_dir+"proc/"


subjs = ["zw","w"]
runs = ["first","second"]

for sub in subjs:
    for run in runs:
        filename = "{dir}{sub}_{run}-raw.fif".format(dir=proc_dir,sub=sub,run=run)
        raw = mne.io.Raw(filename)
        picks = mne.pick_types(raw.info,meg=True,ref_meg=True)
        ica = mne.preprocessing.ICA(n_components=None,allow_ref_meg=True,max_iter=1000)
        ica.fit(raw)
        ica.save("{dir}{sub}_{run}-ica.fif".format(dir=proc_dir,sub=sub,run=run))

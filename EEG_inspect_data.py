#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 17:42:03 2023

@author: arthurlecoz

EEG_inspect_data.py

NOTE TO MYSELF : 
    - raw.info['meas_date'] for pEEG recordings = end of the recording Time
        -> If you wanna grab the start of the recording : need to use the 
            name of the file : ex :
                - UNFILTERED_record-[2023.05.16-10.03.00].edf
                -> raw_unfiltered.info['meas_date'] 
                > datetime.datetime(2023, 5, 16, 10, 41, 23, tzinfo=datetime.timezone.utc)
                => Name of the file :
                    [2023.05.16-10.03.00]

"""
# %% Paths & Packages

import mne
import glob
import numpy as np, pandas as pd
from autoreject import get_rejection_threshold
import config as cfg

from datetime import date
todaydate = date.today().strftime("%d%m%y")

DataType = cfg.DataType
local = cfg.local

if local : 
    local_root_path = "/Users/arthurlecoz/Library/Mobile Documents/com~apple~CloudDocs/Desktop/A_Thesis/2023/Expe/EPISSE"
root_path = local_root_path + "/" + DataType
raw_path = root_path + "/labmeet_mock/Raw"
preproc_path = root_path + "/labmeet_mock/Preproc"
fig_path = root_path + "/labmeet_mock/Figs"

# %% Files

files = glob.glob(root_path + "/*labmeet*/**/*.edf")
filtered_files = glob.glob(root_path + "/*labmeet*/**/FILTERED*.edf")
unfiltered_files = glob.glob(root_path + "/*labmeet*/**/UNFILTER*.edf")

# %%

"""
Weirdly the edf saved by open VIBE are saved in µV directly
but MNE is automatically interpreting the signal as volts
    -> have to do a preprocessing of the files first like 

        raw = mne.io.read_raw_edf(file, preload = True)
        raw.filter(0.1, 40)
        raw.notch_filter([50, 100])
        raw._data = raw._data * 1e-6
        
        mne.export_raw(filename, raw)

"""

raw_unfiltered = mne.io.read_raw_edf(unfiltered_files[1], preload = True)
raw_unfiltered._data = raw_unfiltered._data * 1e-6
raw_unfiltered.filter(0.1, 45, n_jobs = -1)
raw_unfiltered.notch_filter([50, 100], n_jobs = -1)

raw_unfiltered.plot(duration = 30)

# raw_hlf = raw_unfiltered.copy().filter(1, None, n_jobs = -1)

# HERE YOU SHOULD DO A RAW_LIST AND CHANGE EVERYTHING INSIDE OF THE LIST
    # -> Because it's going to be the same thing for both recording
raw_E1 = raw_unfiltered.copy().pick(
    ['E1_O2','E1_C4','E1_F4','E1_Fz','E1_F3','E1_C3','E1_O1','E1_M1']
    )
raw_E2 = raw_unfiltered.copy().pick(
    ['E2_O2','E2_C4','E2_F4','E2_Fz','E2_F3','E2_C3','E2_O1','E2_M1']
    )

raw_E1._data = raw_E1._data - raw_E1._data[-1,:]/2
raw_E2._data = raw_E2._data - raw_E2._data[-1,:]/2

sf = raw_E1.info['sfreq']

allWaves, slowWaves = cfg.SW_detect(raw_E1, fig_path, sf)

# %% 

col_sample = []
col_empty = []
col_event = []

# raw_list = []
epochs_list = []
droplog_l = []

flat_criteria = dict(eeg=1e-6) 

dict_event = {
    "SW/E1_F3" : 1,
    "SW/E1_Fz" : 2,
    "SW/E1_F4" : 3,
    "SW/E1_C3" : 4,
    "SW/E1_C4" : 5,
    "SW/E1_O1" : 6,
    "SW/E1_O2" : 7,
    }

inversed_dict_event = {int(value) : key for key, value in dict_event.items()}

dict_channel_number = {
    "E1_F3" : "1",
    "E1_Fz" : "2",
    "E1_F4" : "3",
    "E1_C3" : "4",
    "E1_C4" : "5",
    "E1_O1" : "6",
    "E1_O2" : "7"
    }

fmt_eve = "%d", "%d", "%d"

channels = ['E1_F3', 'E1_Fz', 'E1_F4', 'E1_C3', 'E1_C4', 'E1_O1', 'E1_O2']

### SW EVENT w/ NAMES

temp_col_sample = []
temp_col_empty = []
temp_col_event = []

for i, channel in enumerate(channels):
    for k, start in enumerate(slowWaves[i][:,0]) :
        temp_col_sample.append(round(start))
        temp_col_empty.append(0)
        temp_col_event.append(int(dict_channel_number[channel]))

temp_event = np.sort(np.c_[
    np.asarray(temp_col_sample),
    np.asarray(temp_col_empty),
    np.asarray(temp_col_event)
    ], axis = 0)

# need to finish by -eve
event_savename = (preproc_path + "/slowwaves_" + todaydate + "-eve.txt")
np.savetxt(event_savename, temp_event, fmt = fmt_eve)

# next logical thingies :
annotations_sw = mne.annotations_from_events(
   temp_event, sf, inversed_dict_event
   )
raw_E1.set_annotations(annotations_sw)

# Epoching
epochs = mne.Epochs(
    raw_E1,
    temp_event, 
    event_id = dict_event, 
    picks = channels,
    tmin = -1,
    tmax = 1,
    baseline = (-1, -0.5),
    preload = True,
    reject = None,
    flat = flat_criteria,
    event_repeated = 'drop'
    )
# Low pass

epochs_hlf = epochs.copy()
epochs.filter(l_freq = None, h_freq = 40, n_jobs = -1)
epochs_hlf.filter(l_freq = 1, h_freq = 40, n_jobs = -1)

# Metadata

nepoch_l = [i for i, epoch in enumerate(epochs)]

new_metadata = pd.DataFrame({
    "n_epoch" : nepoch_l[:len(epochs.events)]
    })

epochs.metadata = new_metadata
epochs_hlf.metadata = new_metadata

# Compute rejection tershold and reject epochs
reject = get_rejection_threshold(epochs_hlf)
epochs_hlf.drop_bad(reject=reject)
epochs_clean = epochs[
    np.isin(
        np.asarray(epochs.metadata.n_epoch),
        np.asarray(epochs_hlf.metadata.n_epoch)
        )
    ] 

droplog_l.append(np.round(epochs_hlf.drop_log_stats(), 2))
print(f"\n{np.round(epochs_hlf.drop_log_stats(), 2)}% of the epochs cleaned were dropped")

# Saving epochs
epochs_savename = preproc_path + "/erp_slowwaves_" + todaydate +"_epo.fif"
epochs_clean.save(
    epochs_savename,
    overwrite = True
    )
annotations_savename = preproc_path + "/erp_slowwaves_epo_annot.csv"
epochs.annotations.save(annotations_savename, overwrite = True)   

# %% Plot
import matplotlib.pyplot as plt

dic_evoked = {}

for i, channel in enumerate(channels) :
    dic_evoked["evoked_" + channel] = epochs_clean["SW/" + channel].average(
        picks = channel)

# for key, value in dic_evoked.items():
#     Chan = key[-2:]
    
#     value.plot(titles = Chan, ylim = dict(eeg=[-10, 16]))

# Create subplots
fig, ((ax1, ax2, ax3), 
      (ax4, ax5, ax6), 
      (ax7, ax8, ax9)) = plt.subplots(
    nrows = 3, 
    ncols = 3,
    figsize = (16,12),
    layout = 'tight'
    )
# Remove box & axis from ax1, ax4, ax9, ax12
ax5.axis('off')
ax8.axis('off')

# Plot your data
dic_evoked['evoked_E1_F3'].plot(
    titles = "F3", axes = ax1)
dic_evoked['evoked_E1_Fz'].plot(
    titles = "Fz", axes = ax2)
dic_evoked['evoked_E1_F4'].plot(
    titles = "F4", axes = ax3)
dic_evoked['evoked_E1_C3'].plot(
    titles = "C3", axes = ax4)
dic_evoked['evoked_E1_C4'].plot(
    titles = "C4", axes = ax6)
dic_evoked['evoked_E1_O1'].plot(
    titles = "O1", axes = ax7)
dic_evoked['evoked_E1_O2'].plot(
    titles = "O2", axes = ax9)

fig_savename = preproc_path + "/swERP_id.png"
fig.savefig(fig_savename, dpi=300)

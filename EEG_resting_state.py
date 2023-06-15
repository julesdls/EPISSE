#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 140623

@author: Arthur_LC

EEG_resting_state.py

"""

# %%% Paths & Packages

import config as cfg
import numpy as np
import mne
import glob
import datetime
from autoreject import get_rejection_threshold
import yasa

root_path = cfg.root_DDE
raw_path = f"{root_path}/CGC_Pilots/Raw"
preproc_path = f"{root_path}/CGC_Pilots/Preproc"
behav_path = f"{root_path}/CGC_Pilots/Behav"
demo_path = f"{root_path}/CGC_Pilots/Demographics"
fig_path = f"{root_path}/CGC_Pilots/Figs"


# %%% Script

all_dates = cfg.recording_dates
alfredo_dates = cfg.alfredo_dates

"""
250523 : 1048 - 1054 | second one ???
010623 : 1029 ? | second one ??? -> last one always?
070623 : 102x ? 
    
    
"""
RS_0 = [
    f"{preproc_path}/Session_1_recording_2_2023.05.25-10_39_38.edf",
    f"{preproc_path}/Session_3_recording_5_2023.06.01-11_03_49.edf",
    f"{preproc_path}/Session_7_recording_2_2023.06.07-10_28_44.edf"
    ]
start_0 = [480, 70, 0]

RS_1 = [
    f"{preproc_path}/Session_1_recording_10_2023.05.25-12_34_57.edf",
    f"{preproc_path}/Session_3_recording_25_2023.06.01-13_23_00.edf",
    f"{preproc_path}/Session_7_recording_19_2023.06.07-13_05_55.edf"
    ]
start_1 = [0, 0, 0]

cait = ["E1"]
cait_ch = [
    'E1_O2','E1_C4','E1_F4','E1_Fz','E1_F3','E1_C3','E1_O1'
    ]
cait_dictch = {
    'E1_O2': 'O2',
    'E1_C4': 'C4',
    'E1_F4': 'F4',
    'E1_Fz': 'Fz',
    'E1_F3': 'F3',
    'E1_C3': 'C3',
    'E1_O1': 'O1' 
        }
jona = ["E2"]
jona_ch = [
    'E2_O2','E2_C4','E2_F4','E2_Fz','E2_F3','E2_C3','E2_O1'
    ]
jona_dictch = {
    'E2_O2': 'O2',
    'E2_C4': 'C4',
    'E2_F4': 'F4',
    'E2_Fz': 'Fz',
    'E2_F3': 'F3',
    'E2_C3': 'C3',
    'E2_O1': 'O1' 
        }

# %% find files

beg_RS = []; 

for i, file in enumerate(RS_0) :
    raw = mne.io.read_raw_edf(file)
    raw = raw.crop(tmin = start_0[i])
    
    cait_raw = raw.copy().pick(cait_ch)
    cait_raw.rename_channels(cait_dictch)
    cait_raw.set_montage("standard_1020")
    cait_epochs = mne.make_fixed_length_epochs(
        cait_raw, duration = 10, preload = True
        )
    reject = get_rejection_threshold(cait_epochs)
    cait_epochs.drop_bad(reject=reject)
    
    jona_raw = raw.copy().pick(jona_ch)
    jona_raw.rename_channels(jona_dictch)
    jona_raw.set_montage("standard_1020")
    jona_epochs = mne.make_fixed_length_epochs(
        jona_raw, duration = 10, preload = True
        )
    reject = get_rejection_threshold(jona_epochs)
    jona_epochs.drop_bad(reject=reject)
    
    beg_RS.append(cait_epochs)
    beg_RS.append(jona_epochs)
    
RS0_epochs = mne.concatenate_epochs(beg_RS)

end_RS = []
for i, file in enumerate(RS_1) :
    raw = mne.io.read_raw_edf(file)
    
    cait_raw = raw.copy().pick(cait_ch)
    cait_raw.rename_channels(cait_dictch)
    cait_raw.set_montage("standard_1020")
    cait_epochs = mne.make_fixed_length_epochs(
        cait_raw, duration = 10, preload = True
        )
    reject = get_rejection_threshold(cait_epochs)
    cait_epochs.drop_bad(reject=reject)
    
    jona_raw = raw.copy().pick(jona_ch)
    jona_raw.rename_channels(jona_dictch)
    jona_raw.set_montage("standard_1020")
    jona_epochs = mne.make_fixed_length_epochs(
        jona_raw, duration = 10, preload = True
        )
    reject = get_rejection_threshold(jona_epochs)
    jona_epochs.drop_bad(reject=reject)
    
    end_RS.append(cait_epochs)
    end_RS.append(jona_epochs)

RS1_epochs = mne.concatenate_epochs(end_RS)


# %% PSD

RS0_epochs.compute_psd(
    method='welch', 
    fmin = 0.5, 
    fmax = 40, 
    n_jobs = -1,
    n_fft = 512,
    n_overlap = 123,
    n_per_seg = 256,
    window = "hamming", 
    average='mean'
    ).plot()
RS0_epochs.plot_psd_topomap()

RS1_epochs.compute_psd(
    method='welch', 
    fmin = 0.5, 
    fmax = 40, 
    n_jobs = -1,
    n_fft = 512,
    n_overlap = 123,
    n_per_seg = 256,
    window = "hamming", 
    average='mean'
    ).plot()
RS1_epochs.plot_psd_topomap()

# %% compute BP - relative

psd_RS0 = RS0_epochs.compute_psd(
    method='welch', 
    fmin = 0.5, 
    fmax = 40, 
    n_jobs = -1,
    n_fft = 512,
    n_overlap = 123,
    n_per_seg = 256,
    window = "hamming", 
    average='mean'
    )

bp_RS0 = yasa.bandpower_from_psd_ndarray(
    np.mean(psd_RS0._data, axis = 0),
    freqs = psd_RS0._freqs,
    relative=True
    )
    
psd_RS1 = RS1_epochs.compute_psd(
    method='welch', 
    fmin = 0.5, 
    fmax = 40, 
    n_jobs = -1,
    n_fft = 512,
    n_overlap = 123,
    n_per_seg = 256,
    window = "hamming", 
    average='mean'
    )

bp_RS1 = yasa.bandpower_from_psd_ndarray(
    np.mean(psd_RS1._data, axis = 0),
    freqs = psd_RS1._freqs,
    relative=True
    )
    
# %% plot BP relative

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

BP = ["Delta", "Theta", "Alpha", "Sigma", "Beta", "Gamma"]
RS_type = [
    "RS0", "RS0", "RS0", "RS0", "RS0", "RS0", 
    "RS1", "RS1", "RS1", "RS1", "RS1", "RS1"
           ]

bp_RS0_av = np.mean(bp_RS0, axis = 1)
bp_RS1_av = np.mean(bp_RS1, axis = 1)

np.concatenate([bp_RS0_av, bp_RS1_av])

df_bp = pd.DataFrame({
    "BP" : BP * 2,
    "values" : np.concatenate([bp_RS0_av, bp_RS1_av]),
    "RS_type" : RS_type
    })

fig, ax = plt.subplots(figsize = (8, 16))
sns.barplot(x = "BP", y = "values", hue = "RS_type", data = df_bp, ax = ax)
ax.set_title("Bandpower - Relative")
ax.set_xlabel("Frequency Band")
ax.set_ylabel("Power")
plt.show(block = False)
plt.savefig(f"{fig_path}/bandpower_relative.png", dpi = 300)
    
# %% plot BP 
bp_RS0 = yasa.bandpower_from_psd_ndarray(
    np.mean(psd_RS0._data, axis = 0),
    freqs = psd_RS0._freqs,
    relative=False
    )
bp_RS1 = yasa.bandpower_from_psd_ndarray(
    np.mean(psd_RS1._data, axis = 0),
    freqs = psd_RS1._freqs,
    relative=False
    )

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

BP = ["Delta", "Theta", "Alpha", "Sigma", "Beta", "Gamma"]
RS_type = [
    "RS0", "RS0", "RS0", "RS0", "RS0", "RS0", 
    "RS1", "RS1", "RS1", "RS1", "RS1", "RS1"
           ]

bp_RS0_av = np.mean(bp_RS0, axis = 1)
bp_RS1_av = np.mean(bp_RS1, axis = 1)

np.concatenate([bp_RS0_av, bp_RS1_av])

df_bp = pd.DataFrame({
    "BP" : BP * 2,
    "values" : np.concatenate([bp_RS0_av, bp_RS1_av]),
    "RS_type" : RS_type
    })

fig, ax = plt.subplots(figsize = (8, 16))
sns.barplot(x = "BP", y = "values", hue = "RS_type", data = df_bp, ax = ax)
ax.set_title("Bandpower - Not Relative")
ax.set_xlabel("Frequency Band")
ax.set_ylabel("Power")
plt.show(block = False)
plt.savefig(f"{fig_path}/bandpower_not_relative.png", dpi = 300)



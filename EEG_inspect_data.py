#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 17:42:03 2023

@author: arthurlecoz

EEG_inspect_data.py

"""
# %% Paths & Packages

import mne
import glob
import numpy as np

local_root_path = "/Users/arthurlecoz/Library/Mobile Documents/com~apple~CloudDocs/Desktop/A_Thesis/2023/Expe/EPISSE"
pilot_path = local_root_path + "/Piloting"
experiment_path = local_root_path + "/Experiment"

# %% Script

files = glob.glob(pilot_path + "/*12_05/*.edf")
filtered_files = glob.glob(pilot_path + "/test_11_05/Raw/FILTERED*.edf")
unfiltered_files = glob.glob(pilot_path + "/test_11_05/Raw/NOFILTER*.edf")

raw_filter = mne.io.read_raw_edf(filtered_files[0], preload = True)
raw_unfiltered = mne.io.read_raw_edf(unfiltered_files[3], preload = True)

raw_unfiltered.plot(duration = 30)
raw_filter.plot(duration=30)

raw = raw_filter.copy().filter(0.1, 40)
raw.notch_filter([50,100])
raw.plot(duration = 30)
raw.compute_psd(method = 'welch', fmin = 0.1, fmax = 40).plot()

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
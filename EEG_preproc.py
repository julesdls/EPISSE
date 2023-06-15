#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 25/05/23

@author: Arthur_LC

EEG_preproc.py

I wanted to concatenate and from x file get only one
but the time resolution is second-precise

The data will be preprocessed and saved with the same beginning name: 
    
    Sess_1_record1_hhmmss.edf
    Sess_1_record2_hhmmss.edf

I will use what I started here to find the following recordings, 
    - process them 
    - and save them

Then whenever I'll do analysis :
    - Load the list of raws
    - Run the sw_det the raws
        - Save only AllWaves
        - From AllWaves compute the mean threshold 
            => (per subject : E1/E2)

    - Add the thought probes events to the raws
    - Epoch before and after the events
        -> Concat the epochs

"""

# %%% Paths & Packages

import config as cfg
import numpy as np
import mne
import glob
import datetime


root_path = cfg.root_DDE

raw_path = f"{root_path}/CGC_Pilots/Raw"
preproc_path = f"{root_path}/CGC_Pilots/Preproc"
behav_path = f"{root_path}/CGC_Pilots/Behav"
demo_path = f"{root_path}/CGC_Pilots/Demographics"

# %%% Script

recording_dates = cfg.recording_dates

for d, date in enumerate(recording_dates) :
    files = glob.glob(f"{raw_path}/*{recording_dates[d]}*/UNFILTERED*.edf")
    start_dates_strings = [
        datetime.datetime.strptime(file.split('/')[-1][19:-5], 
        "%Y.%m.%d-%H.%M.%S").strftime("%Y.%m.%d-%H_%M_%S") for file in files
        ]
    start_dates = [
        datetime.datetime.strptime(
            file.split('/')[-1][19:-5], 
            "%Y.%m.%d-%H.%M.%S").replace(
                tzinfo = datetime.timezone.utc) for file in files
        ]
    
    sorted_dates = sorted(start_dates)
    sorted_indices = sorted(
        range(len(start_dates)), key=lambda i: start_dates[i]
        )
    
    raw_list = [mne.io.read_raw_edf(file, preload = True) for file in files]
    
    for i in range(len(sorted_indices)):
        raw = raw_list[i]                           
        raw._data = raw._data * 1e-6
        raw._data[:8, :] = raw._data[:8,:] - raw._data[7, :]/2
        raw._data[8:16, :] = raw._data[8:16,:] - raw._data[15, :]/2
        
        raw.filter(0.5, 40)
        raw.notch_filter([50, 100])
        
        savename = f"{preproc_path}/Session_{d}_recording_{i}_{start_dates_strings[i]}.edf"
        mne.export.export_raw(
            savename, raw, fmt='edf', overwrite=True
            )
        
        print(f"\n ...Finished processing... {raw_list[i]}")   
    print(f"\nAll files from Session_{d} were processed and saved...")



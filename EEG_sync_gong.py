#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 30_05_23

@author: Arthur_LC
"""

# %%% Paths & Packages

import config as cfg
import numpy as np, pandas as pd
import mne
import glob
import datetime

root_path = cfg.root_DDE
raw_path = f"{root_path}/CGC_Pilots/Raw"
preproc_path = f"{root_path}/CGC_Pilots/Preproc"
behav_path = f"{root_path}/CGC_Pilots/Behav"
psychopy_path = f"{root_path}/CGC_Pilots/Psychopy"
demo_path = f"{root_path}/CGC_Pilots/Demographics"

# %%% Script

# recording_dates = cfg.recording_dates
gong_dates = cfg.gong_dates

for d, date in enumerate(gong_dates) :
    df_gong = pd.read_csv(glob.glob(f"{psychopy_path}/*{date}*/*{date}*.csv")[0])
    files = glob.glob(f"{preproc_path}/Session_{d}_*.edf")
    sorted_files = sorted(files, key=lambda x: int(x.split('_')[5]))
    
    start_dates = [
        datetime.datetime.strptime(
            file[-19:-4], 
            "%y%m%d_%H_%M_%S").replace(
                tzinfo = datetime.timezone.utc) for file in sorted_files
        ]
                
    raw_list = [mne.io.read_raw_edf(file, preload = True) 
                for file in sorted_files]
    end_dates = [raw.info['meas_date'] for raw in raw_list]
    
    for gong_time in df_gong.dateTime.dropna() :
        datetime_gongtime = datetime.datetime.strptime(
            gong_time, "%Y-%m-%d_%H.%M.%S"
            ).replace(tzinfo = datetime.timezone.utc)
        recording_index = cfg.find_recording_for_gong(
            datetime_gongtime, start_dates, end_dates)
        
        if recording_index is not None:
            print(f"The Gong occurred in recording {recording_index+1}")
        else:
            print("The stimulus did not occur within any recording")
            continue
            
        raw = raw_list[recording_index]
        
        gong_onset = (datetime_gongtime - start_dates[recording_index]).total_seconds()
        raw.set_annotations(mne.Annotations(gong_onset, 0, "Gong"))      
        
        savename = f"{preproc_path}/{sorted_files[recording_index].split('/')[-1][:-4]}_Annotated.edf"
        mne.export.export_raw(
            savename, raw, fmt='edf', overwrite=True
            )
        
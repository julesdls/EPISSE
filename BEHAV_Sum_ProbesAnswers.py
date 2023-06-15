#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 30 05 2023

@author: Arthur_LC
"""

# %%% Paths & Packages

import numpy as np, pandas as pd
import glob
import matplotlib.pyplot as plt, seaborn as sns
import config as cfg
import os

if "julissa" in os.getcwd() :
    root_path = cfg.jules_rootpath
    
raw_path = f"{root_path}/CGC_Pilots/Raw"
preproc_path = f"{root_path}/CGC_Pilots/Preproc"
behav_path = f"{root_path}/CGC_Pilots/Behav"
psychopy_path = f"{root_path}/CGC_Pilots/Psychopy"
demo_path = f"{root_path}/CGC_Pilots/Demographics"
fig_path = f"{root_path}/CGC_Pilots/Figs"

# %%% Script

# Hey Julissa was here 

gong_dates = cfg.gong_dates

for date in gong_dates :
    files = glob.glob(f"{behav_path}/*{date}*.csv")
    df_list = [pd.read_csv(file, delimiter = ";") for file in files]
    ID_list = [file[-8:-4] for file in files]
    
    print("\n... Checking if shape is homogeneous among all dfs...")
    for i, ID in enumerate(ID_list):
        print(f"... {ID} report's length is : {len(df_list[i])}")
    answer = input("Everything's Alright?\n(y/n)\n")
    if answer == "n" :
        sys.exit(0)
    
    for i, df in enumerate(df_list) : 
        df.insert(0, "ID", [ID_list[i] for _ in range(len(df))])
        df.insert(0, "n_probe", [f"Probe_{k}" for k in range(len(df))])
        
    
    df_all = pd.concat(df_list)
    grouped_df = df_all.groupby(['n_probe', 'Mindstate'],
                                as_index = False).size()
    grouped_df["size"] = grouped_df["size"].div(len(df_all.ID.unique()))
    
    # grouped_df = df_all.groupby(['n_probe', 'Mindstate']).size().unstack()
    # # Normalize the counts to get proportions
    # proportions_df = grouped_df.div(grouped_df.sum(axis=1), axis=0)
    # # Plot the stacked bar plot
    fig, [ax1, ax2] = plt.subplots(2, 1, figsize = (16, 16))
    # proportions_df.plot(kind='bar', stacked=False, figsize=(10, 6), ax = ax1)
    
    # ax1.set_xlabel('Probe')
    # ax1.set_ylabel('Proportion')
    # ax1.set_title('Proportion of Mindstate for each Probe')
    probe_order = [f"Probe_{i}" for i, _ in enumerate(df_all.n_probe.unique())]
    sns.barplot(data = grouped_df, x = "n_probe", y = 'size', 
                hue = "Mindstate", ax = ax1,
                order = probe_order)
    sns.pointplot(data = df_all, x = "n_probe", y = "Vigilance", ax = ax2)
    ax2.set_title('Vigilance Rating for each Probes')
    
    # Show the plot
    plt.show()
    
    savename = f"{fig_path}/Vigilance_Mindstate_{date}.png"
    plt.savefig(savename, dpi = 300)
    
    





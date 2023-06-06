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
import sys

root_path = cfg.root_DDE
raw_path = f"{root_path}/CGC_Pilots/Raw"
preproc_path = f"{root_path}/CGC_Pilots/Preproc"
behav_path = f"{root_path}/CGC_Pilots/Behav"
psychopy_path = f"{root_path}/CGC_Pilots/Psychopy"
demo_path = f"{root_path}/CGC_Pilots/Demographics"
fig_path = f"{root_path}/CGC_Pilots/Figs"

# %%% Script

inspect = False

gong_dates = cfg.gong_dates
df_l = []
summary_data = {}

for date in gong_dates :
    files = glob.glob(f"{behav_path}/*{date}*.csv")
    df_list = [pd.read_csv(file, delimiter = ";") for file in files]
    ID_list = [file[-8:-4] for file in files]
    
    if inspect: 
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
    grouped_df['size'] = np.asarray(grouped_df.groupby(
        'n_probe')['size'].apply(lambda x: x / x.sum()))
    
    relative_percentage = grouped_df.groupby('Mindstate')['size'].sum() / grouped_df['size'].sum()
    
    summary_data[date] = relative_percentage

    # Plot the simple proportions 
    plt.figure(f"Proportions {date}", figsize = (16, 16))
    sns.barplot(data = grouped_df, x = "Mindstate", y = 'size')
    savename = f"{fig_path}/Proportion_Mindstate_{date}.png"
    plt.savefig(savename, dpi = 300)
    
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
    
    df_l.append(df_all)
plt.close('all')

summary_df = pd.DataFrame(summary_data)
plt.figure("Proportions grandaverage", figsize = (16, 16))
summary_df.reset_index().plot(x='Mindstate', kind='bar', figsize=(10, 6))
plt.xlabel('Mindstate')
plt.ylabel('Relative Percentage')
plt.title('Relative Percentage of Mindstate Globally')
savename = f"{fig_path}/summary_proportion_Mindstate.png"
plt.savefig(savename, dpi = 300)

mean_summary = summary_df.mean(axis=1)
mean_summary.plot(kind='bar', figsize=(10, 6))
plt.xlabel('Mindstate')
plt.ylabel('Mean Relative Percentage')
plt.title('Mean Relative Percentage of Mindstate')
savename = f"{fig_path}/relativeper_proportion_Mindstate.png"
plt.savefig(savename, dpi = 300)



import os
import pandas as pd
from datetime import datetime, timedelta

# Read annot
annot_file = "/wynton/group/andrews/data/HSP/PSG/bids/MGB/sub-S0001112956909/ses-1/eeg/sub-S0001112956909_ses-1_task-psg_annotations.csv"
df_annot = pd.read_csv(annot_file)

# Get resumed time
df_annot['time_dt'] = pd.to_datetime(df_annot['time'], format='%H:%M:%S')
resume_row = df_annot[df_annot['event'] == 'Recording Resumed'].iloc[0]
resumed_time = resume_row['time_dt']
print("Resumed at:", resumed_time)

# Select sleep stages
sleep_mask = df_annot['event'].astype(str).str.startswith('Sleep_stage')
df_sleep = df_annot[sleep_mask].copy()

# Deduce Paused time 
start_time = df_annot.iloc[0]['time_dt']
end_time = df_sleep.iloc[-1]['time_dt'] + timedelta(seconds=30)
annot_duration = (end_time - start_time).total_seconds()
edf_duration = 28110  # seconds, as per your previous calculation
diff = annot_duration - edf_duration 
print("Annotation vs EDF duration difference (s):", diff)

paused_time = start_time + timedelta(seconds=diff)
print("Paused at:", paused_time)


df_sleep_corrected = df_sleep[(df_sleep['time_dt'] < paused_time) | (df_sleep['time_dt'] >= resumed_time)]
print(len(df_sleep_corrected))

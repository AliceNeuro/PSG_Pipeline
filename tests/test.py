# %%
import os
import yasa
import mne
import pandas as pd
from datetime import datetime, timedelta
import warnings
from sklearn.exceptions import InconsistentVersionWarning
import matplotlib.pyplot as plt
import numpy as np
import math

# %%
def get_predicted_stages(row):
    """
    Predict sleep stages using YASA for a single subject. YASA uses EEG, EOG, and EMG channels.
    
    Parameters:
    - edf_file_path (str): Path to the EDF file containing EEG, EOG, and EMG data.
    - age (int): Age of the subject.
    - gender (int): Gender of the subject (0: female, 1: male).
    
    Returns:
    - hypno_pred(list): A list of predicted sleep stages (as integers), corresponding to 30-second epochs.
    """
    edf_file_path = row['edf_path']
    age = row['age']
    gender = row['gender']

    raw = mne.io.read_raw_edf(edf_file_path, preload=False, verbose='Error')

    # Extract channel names for EEG, EOG, and EMG
    # The EEG selected is C4 so using opposite EOG (in this case left EOG)
    channel_labels = raw.ch_names
    emg_name = next((ch for ch in channel_labels if "CHIN" in ch.upper()), None)

    # Try first choice: C4 + E1
    eeg_name = next((ch for ch in channel_labels if "C4" in ch.upper()), None)
    if eeg_name:
        eog_name = next((ch for ch in channel_labels if "E1" in ch.upper()), None)
    else:
        # Fallback: C3 + E2
        eeg_name = next((ch for ch in channel_labels if "C3" in ch.upper()), None)
        if eeg_name:
            eog_name = next((ch for ch in channel_labels if "E2" in ch.upper()), None)

    if not eeg_name or not eog_name or not emg_name:
        print(channel_labels)
        raise ValueError(f"Missing required channels! EEG: {eeg_name}, EOG: {eog_name}, EMG: {emg_name}")
    

    # Predict sleep stages using YASA
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InconsistentVersionWarning)
        sls = yasa.SleepStaging(
            raw,
            eeg_name=eeg_name,
            eog_name=eog_name,
            emg_name=emg_name,
            metadata=dict(age=age, male=gender)
        )
        hypno_pred = sls.predict()
        
    hypno_pred = yasa.hypno_str_to_int(hypno_pred)
    
    return hypno_pred

def plot_hypno_small(predicted_stages, actual_stages):
    import numpy as np
    import math
    import matplotlib.pyplot as plt

    predicted_stages = np.array(predicted_stages)
    actual_stages = np.array(actual_stages)

    # Logical order from bottom to top: N3, N2, N1, R, W
    stage_labels = ['N3', 'N2', 'N1', 'R', 'W']
    mapping = {0:4, 1:2, 2:1, 3:0, 4:3}

    def map_to_plot_order_safe(stages):
        mapped = []
        for s in stages:
            if s is np.nan or (isinstance(s, float) and math.isnan(s)):
                mapped.append(np.nan)
            else:
                mapped.append(mapping[s])
        return np.array(mapped)

    # Map stages
    actual_plot = map_to_plot_order_safe(actual_stages)
    predicted_plot = map_to_plot_order_safe(predicted_stages)

    # --- Single combined plot ---
    plt.figure(figsize=(15, 5))
    plt.step(range(len(actual_plot)), actual_plot, where='mid', color='blue', label='Actual')
    plt.step(range(len(predicted_plot)), predicted_plot, where='mid', 
             color='orange', alpha=0.7, label='Predicted')

    plt.yticks(range(5), stage_labels)
    plt.title('Actual vs Predicted Hypnogram')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_hypno(predicted_stages, actual_stages):
    # Example arrays
    predicted_stages = np.array(predicted_stages)
    actual_stages = np.array(actual_stages)

    # Logical order from bottom to top: N3, N2, N1, R, W
    stage_labels = ['N3', 'N2', 'N1', 'R', 'W']
    # Mapping numeric stage to plotting order
    mapping = {0:4, 1:2, 2:1, 3:0, 4:3}

    def map_to_plot_order_safe(stages):
        mapped = []
        for s in stages:
            if s is np.nan or (isinstance(s, float) and math.isnan(s)):
                mapped.append(np.nan)
            else:
                mapped.append(mapping[s])
        return np.array(mapped)

    # Map stages for plotting
    actual_plot = map_to_plot_order_safe(actual_stages)
    predicted_plot = map_to_plot_order_safe(predicted_stages)

    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)

    # 1. Actual hypnogram
    axes[0].step(range(len(actual_plot)), actual_plot, where='mid', color='blue')
    axes[0].set_yticks(range(5))
    axes[0].set_yticklabels(stage_labels)
    axes[0].set_title('Actual Sleep Stages')
    axes[0].grid(True)

    # 2. Predicted hypnogram
    axes[1].step(range(len(predicted_plot)), predicted_plot, where='mid', color='orange')
    axes[1].set_yticks(range(5))
    axes[1].set_yticklabels(stage_labels)
    axes[1].set_title('Predicted Sleep Stages')
    axes[1].grid(True)

    # 3. Combined hypnogram
    axes[2].step(range(len(actual_plot)), actual_plot, where='mid', color='blue', label='Actual')
    axes[2].step(range(len(predicted_plot)), predicted_plot, where='mid', color='orange', alpha=0.7, label='Predicted')
    axes[2].set_yticks(range(5))
    axes[2].set_yticklabels(stage_labels)
    axes[2].set_title('Combined Hypnogram')
    axes[2].set_xlabel('Epoch')
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()

# %% Select sub ! 
row = pd.Series({
    "edf_path": "/wynton/group/andrews/data/HSP/PSG/bids/MGB/sub-S0001112956909/ses-1/eeg/sub-S0001112956909_ses-1_task-psg_eeg.edf",
    "annot_path" : "/wynton/group/andrews/data/HSP/PSG/bids/MGB/sub-S0001112956909/ses-1/eeg/sub-S0001112956909_ses-1_task-psg_annotations.csv",
    "age": 66,
    "gender": 0, #female
})
# row = pd.Series({
#     "edf_path": "/wynton/group/andrews/data/HSP/PSG/bids/MGB/sub-S0001113071260/ses-1/eeg/sub-S0001113071260_ses-1_task-psg_eeg.edf",
#     "annot_path": "/wynton/group/andrews/data/HSP/PSG/bids/MGB/sub-S0001113071260/ses-1/eeg/sub-S0001113071260_ses-1_task-psg_annotations.csv",
#     "age": 39,
#     "gender": 1, #male
# })

# %% Read annot
df_annot = pd.read_csv(row['annot_path'])


# %% Get actual_stages
df_annot['time_dt'] = pd.to_datetime(df_annot['time'], format='%H:%M:%S')
first_time = df_annot['time_dt'].iloc[0]
df_annot.loc[df_annot['time_dt'] < first_time, 'time_dt'] += pd.Timedelta(days=1)

resume_row = df_annot[df_annot['event'] == 'Recording Resumed'].iloc[0]
resumed_time = resume_row['time_dt']
print("Resumed at:", resumed_time)

# Select sleep stages
sleep_mask = df_annot['event'].astype(str).str.startswith('Sleep_stage')
df_sleep = df_annot[sleep_mask].copy()
stage_dict = {"W": 0, "WAKE": 0, 
                "N1": 1, "1": 1,
                "N2": 2, "2": 2,
                "N3": 3, "N4": 3, "3": 3, "4": 3,
                "R": 4, "REM": 4}

df_sleep['sleep_stage'] = df_sleep['event'].str.replace("Sleep_stage_", "")
stage_dict = {"W": 0, "WAKE": 0, 
                "N1": 1, "1": 1,
                "N2": 2, "2": 2,
                "N3": 3, "N4": 3, "3": 3, "4": 3,
                "R": 4, "REM": 4}
df_sleep['sleep_stage_num'] = df_sleep['sleep_stage'].map(stage_dict)
actual_stages = df_sleep[(df_sleep['time_dt'] > resumed_time)]['sleep_stage_num'].tolist()
len(actual_stages)

# %% Get predicted_stages
predicted_stages = get_predicted_stages(row)
len(predicted_stages)

# %% Plot
max_shift = len(predicted_stages) - len(actual_stages)
for shift in range(max_shift+1):
    after = max_shift - shift
    print(shift, after)
    actual_stages_shifted = [np.nan]*shift + actual_stages + [np.nan]*after
    plot_hypno_small(predicted_stages, actual_stages_shifted)


# %% Look if Recording Resume happen a lot 
master = pd.read_csv("/wynton/group/andrews/data/PSG_Pipeline_Outputs/mastersheets/hsp_bidmc_mastersheet_diagnostic.csv")
#%%
col = "event" # mgb
for idx, row in master.head(200).iterrows():
    annot_file = row["annot_path"]
    if os.path.exists(str(annot_file)):
        df_events = pd.read_csv(annot_file)
        mask = (
            df_events[col].str.contains("Recording", case=False, na=False)
            & ~df_events[col].str.contains("Start|Video|Analyzer", case=False, na=False)
        )
        # If any match, print the full matching rows
        if mask.any():
            print(row["sub_id"], row["session"])
            print(df_events[mask])
# %% OLD CODE

# Deduce Paused time 
start_time = df_annot.iloc[0]['time_dt']
end_time = df_sleep.iloc[-1]['time_dt'] + timedelta(seconds=30)
print(start_time, "-", end_time)
annot_duration = (end_time - start_time).total_seconds()
edf_duration = 28110  # seconds, as per your previous calculation
diff = annot_duration - edf_duration 
print("Annotation vs EDF duration difference (s):", diff)

paused_time = resumed_time - timedelta(seconds=diff)
print("Paused at:", paused_time)


df_sleep_corrected = df_sleep[(df_sleep['time_dt'] < paused_time) | 
                              (df_sleep['time_dt'] > resumed_time)]

print(len(df_sleep_corrected))

# %% Get the actual stages 
actual_stages = df_sleep_corrected['sleep_stage_num'].to_list()
print(len(actual_stages))

# %%

def get_predicted_stages(row):
    """
    Predict sleep stages using YASA for a single subject. YASA uses EEG, EOG, and EMG channels.
    
    Parameters:
    - edf_file_path (str): Path to the EDF file containing EEG, EOG, and EMG data.
    - age (int): Age of the subject.
    - gender (int): Gender of the subject (0: female, 1: male).
    
    Returns:
    - hypno_pred(list): A list of predicted sleep stages (as integers), corresponding to 30-second epochs.
    """
    edf_file_path = row['edf_path']
    age = row['age']
    gender = row['gender']

    raw = mne.io.read_raw_edf(edf_file_path, preload=False, verbose='Error')

    # Extract channel names for EEG, EOG, and EMG
    # The EEG selected is C4 so using opposite EOG (in this case left EOG)
    channel_labels = raw.ch_names
    emg_name = next((ch for ch in channel_labels if "CHIN" in ch.upper()), None)

    # Try first choice: C4 + E1
    eeg_name = next((ch for ch in channel_labels if "C4" in ch.upper()), None)
    if eeg_name:
        eog_name = next((ch for ch in channel_labels if "E1" in ch.upper()), None)
    else:
        # Fallback: C3 + E2
        eeg_name = next((ch for ch in channel_labels if "C3" in ch.upper()), None)
        if eeg_name:
            eog_name = next((ch for ch in channel_labels if "E2" in ch.upper()), None)

    if not eeg_name or not eog_name or not emg_name:
        print(channel_labels)
        raise ValueError(f"Missing required channels! EEG: {eeg_name}, EOG: {eog_name}, EMG: {emg_name}")
    

    # Predict sleep stages using YASA
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InconsistentVersionWarning)
        sls = yasa.SleepStaging(
            raw,
            eeg_name=eeg_name,
            eog_name=eog_name,
            emg_name=emg_name,
            metadata=dict(age=age, male=gender)
        )
        hypno_pred = sls.predict()
        
    hypno_pred = yasa.hypno_str_to_int(hypno_pred)
    
    return hypno_pred



# %%
row = pd.Series({
    "edf_path": "/wynton/group/andrews/data/HSP/PSG/bids/MGB/sub-S0001112956909/ses-1/eeg/sub-S0001112956909_ses-1_task-psg_eeg.edf",
    "age": 66,
    "gender": 0, #female
})
row = pd.Series({
    "edf_path": "/wynton/group/andrews/data/HSP/PSG/bids/MGB/sub-S0001113071260/ses-1/eeg/sub-S0001113071260_ses-1_task-psg_eeg.edf",
    "age": 39,
    "gender": 1, #male
})
predicted_stages = get_predicted_stages(row)
len(predicted_stages)
# %%
import matplotlib.pyplot as plt
import numpy as np
import math

def plot_hypno(predicted_stages, actual_stages):
    # Example arrays
    predicted_stages = np.array(predicted_stages)
    actual_stages = np.array(actual_stages)

    # Logical order from bottom to top: N3, N2, N1, R, W
    stage_labels = ['N3', 'N2', 'N1', 'R', 'W']
    # Mapping numeric stage to plotting order
    mapping = {0:4, 1:2, 2:1, 3:0, 4:3}

    def map_to_plot_order_safe(stages):
        mapped = []
        for s in stages:
            if s is np.nan or (isinstance(s, float) and math.isnan(s)):
                mapped.append(np.nan)
            else:
                mapped.append(mapping[s])
        return np.array(mapped)

    # Map stages for plotting
    actual_plot = map_to_plot_order_safe(actual_stages)
    predicted_plot = map_to_plot_order_safe(predicted_stages)

    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)

    # 1. Actual hypnogram
    axes[0].step(range(len(actual_plot)), actual_plot, where='mid', color='blue')
    axes[0].set_yticks(range(5))
    axes[0].set_yticklabels(stage_labels)
    axes[0].set_title('Actual Sleep Stages')
    axes[0].grid(True)

    # 2. Predicted hypnogram
    axes[1].step(range(len(predicted_plot)), predicted_plot, where='mid', color='orange')
    axes[1].set_yticks(range(5))
    axes[1].set_yticklabels(stage_labels)
    axes[1].set_title('Predicted Sleep Stages')
    axes[1].grid(True)

    # 3. Combined hypnogram
    axes[2].step(range(len(actual_plot)), actual_plot, where='mid', color='blue', label='Actual')
    axes[2].step(range(len(predicted_plot)), predicted_plot, where='mid', color='orange', alpha=0.7, label='Predicted')
    axes[2].set_yticks(range(5))
    axes[2].set_yticklabels(stage_labels)
    axes[2].set_title('Combined Hypnogram')
    axes[2].set_xlabel('Epoch')
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()

# %%
plot_hypno(predicted_stages, actual_stages)
# %%
df_sleep[(df_sleep['time_dt'] >= resumed_time)]['sleep_stage_num']

# %%
actual_stages = df_sleep[(df_sleep['time_dt'] > resumed_time)]['sleep_stage_num'].tolist()
print(actual_stages[10:30])
actual_stages = [np.nan]*10 + actual_stages
len(actual_stages)

# %%
plot_hypno(predicted_stages, actual_stages)

# %%
print(predicted_stages[-10:])
actual_stages = df_sleep[(df_sleep['time_dt'] > resumed_time)]['sleep_stage_num'].tolist()
actual_stages = [np.nan]*15 + actual_stages[:-5]

# %%
plot_hypno(predicted_stages, actual_stages)

# %%

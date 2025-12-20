import numpy as np
from collections import Counter

def extract_aasm(full_sleep_stages, sleep_stages, df_events, row):
    sfreq_global = row['sfreq_global']
    psg_id = f"sub-{row['sub_id']}_ses-{row['session']}"
    metrics = compute_sleep_metrics(full_sleep_stages, sleep_stages, df_events, sfreq_global, psg_id)
    indices = compute_event_indices(
        full_sleep_stages, 
        sfreq_global, 
        df_events,
        tst=metrics['TST'],
        n1_pct=metrics['N1%'],
        n2_pct=metrics['N2%'],
        n3_pct=metrics['N3%'],
        rem_pct=metrics['REM%']
    )       
    combined = {**metrics, **indices}

    return combined

def compute_sleep_metrics(full_sleep_stages, sleep_stages, df_events, sfreq_global, psg_id):

    # 1. Cut night period
    lights_sleep_stages = cut_lights_off_on(full_sleep_stages, df_events, sfreq_global, psg_id)

    # 2. Sleep architecture
    tst = total_sleep_time(lights_sleep_stages, sfreq_global)       # in hours
    trt = total_recording_time(lights_sleep_stages, sfreq_global)   # in hours
    se = sleep_efficiency(tst, trt)
    n1_pct = stage_percentage(tst, sleep_stages, sfreq_global, 1)
    n2_pct = stage_percentage(tst, sleep_stages, sfreq_global, 2)
    n3_pct = stage_percentage(tst, sleep_stages, sfreq_global, 3)
    rem_pct = stage_percentage(tst, sleep_stages, sfreq_global, 4)
    waso_val = waso(sleep_stages, sfreq_global)
    sl = sleep_latency(lights_sleep_stages, sfreq_global)
    rl = rem_latency(sleep_stages, sfreq_global)
    sfi = sleep_fragmentation_index(tst, sleep_stages)

    metrics = {
        'TST': float(tst) ,
        'TRT': float(trt),
        'SE': float(se),
        'REM%': float(rem_pct),
        'N1%': float(n1_pct),
        'N2%': float(n2_pct),
        'N3%': float(n3_pct),
        'WASO': float(waso_val),
        'SL': float(sl),
        'RL': float(rl),
        'SFI': float(sfi)
    }

    return metrics


def seconds_to_minutes(sec):
    return sec / 60

def seconds_to_hours(sec):
    return sec / 3600

# Helper: sleep mask for 0=wake, 1-4 sleep, ignore NaN
def sleep_mask(arr):
    return np.isin(arr, [1,2,3,4])

# -----------------------------
# Cut night period based on lights off/on events
# -----------------------------
def cut_lights_off_on(full_sleep_stages, df_events, sfreq_global, psg_id):
    df_events['event_type'] = df_events['event_type'].fillna('').astype(str)
    light_events = df_events[df_events['event_type'].str.contains('light', case=False)]

    # --- Lights off ---
    off_events = light_events[
        light_events['event_type'].str.contains('out|off', case=False)
    ].copy()
    off_events = off_events[off_events['onset'].notna()]

    if len(off_events) > 1:
        print(f"[WARNING] {psg_id}: Multiple lights-off events detected ({len(off_events)}). Using the first one.")

    # --- Lights on ---
    on_events = light_events[
        light_events['event_type'].str.contains('on', case=False)
    ].copy()
    on_events = on_events[on_events['onset'].notna()]

    if len(on_events) > 1:
        print(f"[WARNING] {psg_id}: Multiple lights-on events detected ({len(on_events)}). Using the first one.")

    # --- Determine indices ---
    if len(off_events) > 0 and len(on_events) > 0:
        off_idx = int(off_events['onset'].min() * sfreq_global)
        on_idx  = int(on_events['onset'].max() * sfreq_global)
        lights_sleep_stages = full_sleep_stages[off_idx:on_idx]
    else:
        # If any of the events is missing, return the full signal
        lights_sleep_stages = full_sleep_stages

    return lights_sleep_stages


def cut_lights_off_on(full_sleep_stages, df_events, sfreq_global, psg_id):
    light_events = df_events[df_events['event_type'].fillna('').astype(str).str.contains('light', case=False)]

    off_events = light_events[light_events['event_type'].str.contains('out|off', case=False, na=False)]
    on_events  = light_events[light_events['event_type'].str.contains('on', case=False, na=False)]

    if len(off_events) > 1:
        print(f"[WARNING] {psg_id}: multiple lights-off events ({len(off_events)})")
    if len(on_events) > 1:
        print(f"[WARNING] {psg_id}: multiple lights-on events ({len(on_events)})")

    if len(off_events) > 0 and len(on_events) > 0:
        off_idx = int(off_events['onset'].min() * sfreq_global)
        on_idx  = int(on_events['onset'].max() * sfreq_global)
        lights_sleep_stages = full_sleep_stages[off_idx:on_idx]
    else:
        lights_sleep_stages = full_sleep_stages

    return lights_sleep_stages

# -----------------------------
# Sleep architecture metrics
# -----------------------------
def total_sleep_time(lights_sleep_stages, sfreq_global): 
    mask = sleep_mask(lights_sleep_stages)
    if mask is None or np.sum(mask) == 0:
        return 0
    return seconds_to_hours(np.sum(mask)/sfreq_global)

def total_recording_time(lights_sleep_stages, sfreq_global):
    return seconds_to_hours(len(lights_sleep_stages)/sfreq_global)

def sleep_efficiency(tst, trt):
    return (tst / trt) * 100 if trt>0 else np.nan

def stage_percentage(tst, sleep_stages, sfreq_global, stage):
    if tst==0:
        return np.nan
    stage_sec = np.sum(sleep_stages==stage)/sfreq_global
    return stage_sec / (tst*3600) * 100  # percent of TST

def waso(sleep_stages, sfreq_global):
    wake_after = np.sum(sleep_stages==0)
    return seconds_to_minutes(wake_after / sfreq_global)

def sleep_latency(lights_sleep_stages, sfreq_global):
    mask = sleep_mask(lights_sleep_stages)
    if mask is None or np.sum(mask) == 0:
        return np.nan 
    first_sleep_idx = np.argmax(mask)
    return seconds_to_minutes(first_sleep_idx / sfreq_global)

def rem_latency(sleep_stages, sfreq_global):
    rem_indices = np.where(sleep_stages == 4)[0]
    if rem_indices.size == 0:
        return np.nan
    first_rem_idx = rem_indices[0]
    return seconds_to_minutes(first_rem_idx / sfreq_global)

def sleep_fragmentation_index(tst, sleep_stages):
    """
    Counts specific transitions per hour of sleep.
        Case A: any stage -> Wake (0)
        Case B: NREM or REM sleep (2,3,4) -> Light sleep (1)
    """
    if tst==0:
        return np.nan
    
    # Remove NaNs
    stages = sleep_stages[~np.isnan(sleep_stages)]
    if len(stages) < 2 or tst == 0:
        return np.nan

    # create masks to deduce vlaid transitions
    prev = stages[:-1]
    next_ = stages[1:]
    transitions = np.isin(prev, [2, 3, 4]) & np.isin(next_, [0, 1])
    num_transitions = np.sum(transitions)

    # Per hour of sleep
    return num_transitions / tst

# -----------------------------
# Respiratory & event indices
# -----------------------------
def compute_event_indices(full_sleep_stages, sfreq_global, df_events, tst, n1_pct, n2_pct, n3_pct, rem_pct):
    if tst == 0:
        return {k: np.nan for k in ['AHI', 'AHI_NREM', 'AHI_REM', 'RDI','OAI',
                                    'CAI','MAI', 'HyI', 'RERAI', 'LMI', 'PLMI', 'ArI']}

    # Convert event_type for only sleep period to lowercase for consistency
    mask = sleep_mask(full_sleep_stages) & ~np.isnan(full_sleep_stages)
    if mask is None or np.sum(mask)==0:
        df_sleep = df_events.iloc[0:0].copy()
    else:
        sleep_start_idx = np.argmax(mask)          
        sleep_end_idx   = len(mask) - 1 - np.argmax(mask[::-1])  
        print(sleep_start_idx, sleep_end_idx)
        start_time = sleep_start_idx / sfreq_global
        end_time   = sleep_end_idx / sfreq_global

        # Include any event overlapping sleep
        df_sleep = df_events[
            (df_events['onset'] <= end_time) &  
            (df_events['onset'] + df_events['duration'] > start_time)
        ].copy()

    events = df_sleep['event_type'].str.lower().fillna('')

    # Count occurrences by keywords
    count_apnea = np.sum(events.str.contains('apnea', case=False, na=False))
    count_hypopnea = np.sum(
        events.str.contains('hypopnea', case=False, na=False))
    count_obstructive = np.sum(
        events.str.contains('obstruct', case=False, na=False) & 
        events.str.contains('apnea', case=False, na=False))
    count_central = np.sum(
        events.str.contains('central', case=False, na=False) & 
        events.str.contains('apnea', case=False, na=False))
    count_mixed = np.sum(events.str.contains('mix', case=False, na=False) & 
                         events.str.contains('apnea', case=False, na=False))
    count_rera = np.sum(events.str.contains('rera', case=False, na=False))
    count_arousal = np.sum(events.str.contains('arousal', case=False, na=False))
    count_limb_mov = np.sum(
        (events.str.contains('limb|leg|periodic', case=False, na=False) & 
         events.str.contains('movement', case=False, na=False)) |
        events.str.contains(r'\[lm\]', case=False, na=False) |
        events.str.contains('plm', case=False, na=False)
    )
    count_plm = np.sum(
        (events.str.contains('periodic', case=False, na=False) & 
         events.str.contains('movement', case=False, na=False)) |
        (events.str.contains('plm', case=False, na=False) &
         ~events.str.contains('isolated', case=False, na=False))
    )

    # Compute indices
    AHI = (count_apnea + count_hypopnea) / tst
    RDI = (count_apnea + count_hypopnea + count_rera) / tst
    OAI = count_obstructive / tst
    CAI = count_central / tst
    MAI = count_mixed / tst
    HyI = count_hypopnea / tst
    RERAI = count_rera / tst
    LMI = count_limb_mov / tst
    PLMI = count_plm / tst
    ArI = (count_arousal + count_rera) / tst
    
    # Compute AHI_NREM and AHI_REM
    df_pnea = df_events[df_events['event_type'].str.lower().fillna('').str.contains('pnea')].copy()
    n_samples = len(full_sleep_stages)

    # Ensure the column exists
    if 'sleep_stage' not in df_pnea.columns:
        df_pnea['sleep_stage'] = np.nan

    for idx, row in df_pnea.iterrows():
        # Compute indices
        start_idx = int(row['onset'] * sfreq_global)
        end_idx   = int((row['onset'] + row['duration']) * sfreq_global)

        # Clip to valid range
        start_idx = max(0, min(start_idx, n_samples-1))
        end_idx   = max(start_idx+1, min(end_idx, n_samples))  # ensure at least 1 sample

        # Extract sleep segment
        seg = full_sleep_stages[start_idx:end_idx]
        seg = seg[~np.isnan(seg)]
        if len(seg) == 0:
            continue  # skip if all NaN

        # Find dominant sleep stage
        values, counts = np.unique(seg, return_counts=True)
        dominant_stage = values[np.argmax(counts)]
        df_pnea.at[idx, 'sleep_stage'] = dominant_stage

    # for idx, row in df_pnea.iterrows():
    #     # Index at the onset of the event
    #     onset_idx = int(row['onset'] * sfreq_global)
    #     onset_idx = max(0, min(n_samples - 1, onset_idx))  # Ensure within bounds
    #     # Take the sleep stage at onset
    #     sleep_stage_at_onset = full_sleep_stages[onset_idx]
    #     # Skip if NaN
    #     if np.isnan(sleep_stage_at_onset):
    #         continue
    #     df_pnea.at[idx, 'sleep_stage'] = sleep_stage_at_onset

    # for idx, row in df_pnea.iterrows():
    #     # Index at the offset (end) of the event
    #     offset_idx = int((row['onset'] + row['duration']) * sfreq_global)
    #     offset_idx = max(0, min(n_samples - 1, offset_idx))  # Ensure within bounds
    #     sleep_stage_at_offset = full_sleep_stages[offset_idx]
    #     # Skip if NaN
    #     if np.isnan(sleep_stage_at_offset):
    #         continue
    #     df_pnea.at[idx, 'sleep_stage'] = sleep_stage_at_offset  
    
    pnea_nrem = np.sum(df_pnea['sleep_stage'].isin([1, 2, 3]))
    pnea_rem  = np.sum(df_pnea['sleep_stage'] == 4)
    nrem_hours = ((n1_pct + n2_pct + n3_pct) / 100) * tst
    rem_hours  = (rem_pct / 100) * tst
    AHI_NREM = pnea_nrem / nrem_hours if nrem_hours > 0 else np.nan
    AHI_REM  = pnea_rem / rem_hours if rem_hours > 0 else np.nan

    indices = {
        'AHI': AHI,
        'AHI_NREM': AHI_NREM,
        'AHI_REM': AHI_REM,
        'RDI': RDI,
        'OAI': OAI,
        'CAI': CAI,
        'MAI': MAI,
        'HyI': HyI,
        'RERAI': RERAI,
        'LMI': LMI,
        'PLMI': PLMI,
        'ArI': ArI,
    }

    return indices

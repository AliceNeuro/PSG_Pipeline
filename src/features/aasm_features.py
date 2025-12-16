import numpy as np

def extract_aasm(full_sleep_stages, sleep_stages, df_events, row):
    sfreq_global = row['sfreq_global']
    psg_id = f"sub-{row['sub_id']}_ses-{row['session']}"
    metrics = compute_sleep_metrics(full_sleep_stages, sleep_stages, df_events, sfreq_global, psg_id)
    indices = compute_event_indices(
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
    # Filter light events
    light_events = df_events[df_events['event_type'].str.contains('light', case=False, na=False)]

    # Find lights off
    off_events = light_events[light_events['event_type'].str.contains('out|off', case=False, na=False)]
    if len(off_events) > 1:
        print(f"[WARNING] {psg_id}: Lights off event length: {len(off_events)}")
    off_idx = int(off_events['onset'].iloc[0] * sfreq_global) if len(off_events) > 0 else None

    # Find lights on
    on_events = light_events[light_events['event_type'].str.contains('on', case=False, na=False)]
    if len(on_events) > 1:
        print(f"[WARNING] {psg_id}: Lights on event length: {len(on_events)}")
    on_idx = int(on_events['onset'].iloc[0] * sfreq_global) if len(on_events) > 0 else None

    # Cut sleep stages based on lights off/on
    if off_idx is not None and on_idx is not None:
        lights_sleep_stages = full_sleep_stages[off_idx:on_idx]
    else:
        lights_sleep_stages = full_sleep_stages

    return lights_sleep_stages

# -----------------------------
# Sleep architecture metrics
# -----------------------------
def total_sleep_time(lights_sleep_stages, sfreq_global): 
    mask = sleep_mask(lights_sleep_stages)
    return seconds_to_hours(np.sum(mask)/sfreq_global)

def total_recording_time(lights_sleep_stages, sfreq_global):
    # Should I include Nan ?
    # valid = ~np.isnan(lights_sleep_stages)
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

def sleep_latency(light_sleep_stages, sfreq_global):
    mask = sleep_mask(light_sleep_stages)
    if not np.any(mask) is None:
        return np.nan
    first_sleep_idx = np.argmax(mask)
    return seconds_to_minutes(first_sleep_idx / sfreq_global)

def rem_latency(sleep_stages, sfreq_global):
    rem_idx = np.where(sleep_stages == 4)
    if len(rem_idx) == 0:
        return np.nan
    first_rem = rem_idx[0]
    return seconds_to_minutes(first_rem / sfreq_global)

def sleep_fragmentation_index(tst, sleep_stages):
    # Remove NaNs
    stages = sleep_stages[~np.isnan(sleep_stages)]
    if len(stages) < 2:
        return np.nan
    
    # Identify deep to light/wake transitions
    deep = (stages[:-1] == 2) | (stages[:-1] == 3)
    shallow_or_wake = (stages[1:] == 0) | (stages[1:] == 1)
    transitions = deep & shallow_or_wake
    num_transitions = np.sum(transitions)

    # SFI = wake transitions per hour of sleep
    if tst == 0:
        return np.nan

    return num_transitions / tst

# -----------------------------
# Respiratory & event indices
# -----------------------------
def compute_event_indices(df_events, tst, n1_pct, n2_pct, n3_pct, rem_pct):
    if tst == 0 or np.isnan(tst):
        return {k: np.nan for k in ['AHI', 'AHI_NREM', 'AHI_REM', 'RDI','OAI','CAI',
                                    'MAI', 'HyI', 'RERAI', 'LMI', 'PLMI', 'ArI']}

    # Convert event_type to lowercase for consistency
    events = df_events['event_type'].str.lower().fillna('')

    # Count occurrences by keywords
    count_apnea = np.sum(events.str.contains('apnea', case=False, na=False))
    count_hypopnea = np.sum(events.str.contains('hypopnea', case=False, na=False))
    count_obstructive = np.sum(events.str.contains('obstructive', case=False, na=False) & 
                         events.str.contains('apnea', case=False, na=False))
    count_central = np.sum(events.str.contains('central', case=False, na=False) & 
                         events.str.contains('apnea', case=False, na=False))
    count_mixed = np.sum(events.str.contains('mixed', case=False, na=False) & 
                         events.str.contains('apnea', case=False, na=False))
    count_rera = np.sum(events.str.contains('rera', case=False, na=False))
    count_arousal = np.sum(events.str.contains('arousal', case=False, na=False))
    count_limb_mov = np.sum(
        (events.str.contains('limb|leg|arm', case=False, na=False) & 
         events.str.contains('movement', case=False, na=False)) |
        events.str.contains(r'\[lm\]', case=False, na=False) |
        (events.str.contains('period', case=False, na=False) & 
         events.str.contains('movement', case=False, na=False)) |
        events.str.contains('plm', case=False, na=False)
    )
    count_plm = np.sum(
        (events.str.contains('period', case=False, na=False) & 
         events.str.contains('movement', case=False, na=False)) |
        events.str.contains('plm', case=False, na=False) 
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
    df_pnea = df_events[df_events['event_type'].str.lower().fillna('').str.contains('pnea')]
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

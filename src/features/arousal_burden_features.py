import numpy as np
import pandas as pd

def extract_ab(row, df_events, sleep_stages, sleep_onset_offset_sec):

    sfreq_global = row["sfreq_global"]

    # Keep only arousals and RERA
    if "event_type" in df_events.columns:
        df_arousals = df_events[
            df_events["event_type"].str.contains("arousal|rera", case=False, na=False)
        ].copy() # Adding rera events as arousals
    else: 
        df_arousals = pd.DataFrame()

    # Drop useless columns if present
    for col in ["channel", "lowest_spo2", "desaturation"]:
        if col in df_arousals.columns:
            df_arousals = df_arousals.drop(columns=[col])

    if df_arousals.empty:
        # Return zeros for all 5 categories × 3 metrics
        suffixes = ["SLEEP", "N1", "N2", "N3", "N2N3", "NREM", "REM"]
        return {
            f"total_min@{s}": 0.0
            for s in suffixes
        } | {
            f"arousal_min@{s}": np.nan
            for s in suffixes
        } | {
            f"arousal_burden@{s}": np.nan
            for s in suffixes
        } | {
            f"arousal_index@{s}": np.nan
            for s in suffixes
        }

    # Stage definitions
    STAGE_SLEEP = [1, 2, 3, 4] 
    STAGE_N1  = [1]
    STAGE_N2  = [2]
    STAGE_N3  = [3]
    STAGE_N2N3= [2, 3]
    STAGE_NREM= [1, 2, 3]
    STAGE_REM = [4]

    results = {}

    # ---- Call sub-function for each condition ----
    results |= compute_arousal_burden_for_stage("SLEEP", sfreq_global, sleep_stages, STAGE_SLEEP, df_arousals, sleep_onset_offset_sec)
    results |= compute_arousal_burden_for_stage("N1",    sfreq_global, sleep_stages, STAGE_N1, df_arousals, sleep_onset_offset_sec)
    results |= compute_arousal_burden_for_stage("N2",    sfreq_global, sleep_stages, STAGE_N2, df_arousals, sleep_onset_offset_sec)
    results |= compute_arousal_burden_for_stage("N3",    sfreq_global, sleep_stages, STAGE_N3, df_arousals, sleep_onset_offset_sec)
    results |= compute_arousal_burden_for_stage("N2N3",  sfreq_global, sleep_stages, STAGE_N2N3, df_arousals, sleep_onset_offset_sec)
    results |= compute_arousal_burden_for_stage("NREM",  sfreq_global, sleep_stages, STAGE_NREM, df_arousals, sleep_onset_offset_sec)
    results |= compute_arousal_burden_for_stage("REM",   sfreq_global, sleep_stages, STAGE_REM, df_arousals, sleep_onset_offset_sec)

    return results



def compute_arousal_burden_for_stage(suffix, sfreq_global, sleep_stages, stage_codes, df_arousals, sleep_onset_offset_sec):
    """Compute sleep minutes, arousal minutes, and arousal burden for given stage codes."""

    # -------- Sleep time for these stages --------
    TST_samples = np.sum(np.isin(sleep_stages, stage_codes))
    TST_min = TST_samples / sfreq_global / 60.0
    
    if TST_min == 0:
        return {
            f"total_min@{suffix}": TST_min,
            f"arousal_min@{suffix}": np.nan,
            f"arousal_burden@{suffix}": np.nan,
            f"arousal_index@{suffix}": np.nan
        }

    # -------- Filter arousals occurring during these stages --------
    arousal_during_sleep = []
    for _, event in df_arousals.iterrows():  
        # Convert to sample indices (not clipped yet)
        start_idx = int((event["onset"] - sleep_onset_offset_sec) * sfreq_global)
        end_idx   = int((event["onset"] + event["duration"] - sleep_onset_offset_sec) * sfreq_global)

        # Clip index range
        start_idx = max(0, min(start_idx, len(sleep_stages) - 1))
        end_idx   = max(0, min(end_idx,   len(sleep_stages) - 1))


        if np.any(np.isin(sleep_stages[start_idx:end_idx + 1], stage_codes)):
            arousal_during_sleep.append(event)

    if not arousal_during_sleep:
        return {
            f"total_min@{suffix}": TST_min,
            f"arousal_min@{suffix}": 0.0,
            f"arousal_burden@{suffix}": 0.0,
            f"arousal_index@{suffix}": 0.0 
        }

    # -------- Merge overlapping arousals --------
    df_stage = pd.DataFrame(arousal_during_sleep).sort_values("onset").reset_index(drop=True)

    merged = []
    current_start = None
    current_end = None

    for _, row in df_stage.iterrows():
        start = row["onset"]
        end = row["onset"] + row["duration"]

        if current_start is None:
            current_start = start
            current_end = end
        else:
            if start <= current_end:
                current_end = max(current_end, end)
            else:
                merged.append({"onset": current_start, "duration": current_end - current_start})
                current_start = start
                current_end = end

    if current_start is not None:
        merged.append({"onset": current_start, "duration": current_end - current_start})

    df_merged = pd.DataFrame(merged)

    # -------- Fix offset in wake -------- 
    corrected_onsets = []
    corrected_durations = []

    for _, row in df_merged.iterrows():
        onset_sec = row["onset"]
        end_sec   = row["onset"] + row["duration"]

        # Convert to sample indices (not clipped yet)
        start_idx = int((onset_sec - sleep_onset_offset_sec) * sfreq_global)
        end_idx   = int((end_sec - sleep_onset_offset_sec) * sfreq_global)

        # Clip to valid range
        start_idx = max(0, min(start_idx, len(sleep_stages)-1))
        end_idx   = max(0, min(end_idx,   len(sleep_stages)-1))

        # If end in wake → cut to last sleep sample
        if sleep_stages[end_idx] == 0:  # wake code
            i = end_idx
            while i >= 0 and sleep_stages[i] == 0:   # wake code
                i -= 1

            if i >= 0:
                # extend to end of last sleep sample = (i+1)
                new_end_sec = (i + 1) / sfreq_global + sleep_onset_offset_sec
            else:
                # the entire thing before onset is already wake → leave unchanged
                new_end_sec = end_sec
        else:
            new_end_sec = end_sec

        corrected_onsets.append(onset_sec)
        corrected_durations.append(new_end_sec - onset_sec)

    df_merged["duration"] = corrected_durations

    # -------- Arousal duration --------
    N = len(df_merged) # Number of arousals (after merging)
    arousal_min = df_merged["duration"].sum() / 60.0
    arousal_burden = (arousal_min / TST_min) * 100.0
    arousal_index = N / TST_min * 60.0  # arousals per hour

    return {
        f"total_min@{suffix}": TST_min,
        f"arousal_min@{suffix}": arousal_min,
        f"arousal_burden@{suffix}": arousal_burden,
        f"arousal_index@{suffix}": arousal_index,
    }
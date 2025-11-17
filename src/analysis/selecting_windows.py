import numpy as np
from datetime import datetime, timedelta, timezone
from collections import Counter

def selecting_windows(config, row, sleep_stages, sleep_onset_time, processed_signals):
    """
    Select windows for ECG and EMG signals based on the provided configuration.

    Parameters
    ----------
    config : dict
        Configuration dictionary for window selection.
    epoch_length_sec : int
        Epoch length in seconds.
    sleep_stages : list
        List of sleep stages.
    sleep_onset_time : datetime
        Sleep onset time.
    processed_signals : dict
        Dictionary containing processed signal data.

    Returns
    -------
    windows_dict : dict
        Dictionary containing selected windows for each signal.
    """
    windows_dict = {}

    # Iterate over all processed signals
    for signal_name, signal_data in processed_signals.items():
        windows = None  # Initialize windows variable
        
        # Handle ECG signals
        if signal_name.upper().startswith("ECG"):
            if signal_data.get("clean_rpeaks_mask") is not None:
                if config.run.verbose:
                    print(f"[INFO] Selecting window for signal {signal_name}")
                windows = selecting_windows_ecg(
                    config, row, sleep_stages, sleep_onset_time, signal_data
                )
            else:
                continue

        # Handle EMG signals
        elif signal_name.upper().startswith("EMG"):
            windows = selecting_windows_emg(
                config, sleep_stages, sleep_onset_time, signal_data
            )
        
        # Handle unsupported signal types
        else:
            print(f"[WARNING] Sub {row['sub_id']}: No window selection defined for signal: {signal_name}")
            continue

        if windows is not None:
            windows_dict[signal_name] = windows
        else:
            if config.run.verbose:
                print(f"[WARNING] Sub {row['sub_id']}: No window selection for signal {signal_name}")

    return windows_dict


def selecting_windows_ecg(config, row, sleep_stages, sleep_onset_time, signal_data):
    """
    Selects non-overlapping, clean ECG windows with sufficient sleep stage purity.
    Returns a dictionary of windows grouped by sleep stage and purity.
    """
    # Extract ECG-related signals

    sfreq = signal_data["sfreq_signal"]
    clean_rpeaks_mask = signal_data["clean_rpeaks_mask"]

    # Config-driven window size (exemple for 5min)
    window_size_min = config.analysis.window_size_min
    window_size_sec = window_size_min * 60 # 300sec 
    window_size_samples = int(window_size_sec * sfreq) # 153'600 samples
    
    # Config-driven window step (exemple for 1 epoch = 30sec)
    window_step_sec = int(config.analysis.window_step_sec)
    window_step_samples = int(window_step_sec * sfreq) 

    max_sample_start = len(clean_rpeaks_mask) - window_size_samples
    sample_starts = np.arange(0, max_sample_start + 1, window_step_samples)

    if config.run.verbose:
        print(f"Max sample start: {max_sample_start}")
        print(f"Total candidate windows: {len(sample_starts)}")

    # Gather valid (clean) window candidates
    window_candidates = []
    for start_idx in sample_starts:
        end_idx = start_idx + window_size_samples
    
        if end_idx > len(clean_rpeaks_mask):
            continue
        
        clean_ratio = clean_rpeaks_mask[start_idx:end_idx].mean()
        if clean_ratio >= config.analysis.clean_rpeaks_ratio_threshold: 
            stage_window = sleep_stages[start_idx:end_idx]
            window_candidates.append((start_idx, stage_window))
        else:
            if config.run.verbose:
                time = sleep_onset_time + timedelta(seconds=int(start_idx/sfreq))
                # print(f"Rejected window {time} ({start_idx}) due to low clean ratio:", clean_ratio)
    if config.run.verbose:
        print(f"Clean ECG windows: {len(window_candidates)}")
    if len(window_candidates) == 0:
        return None

    # Stage-specific selection with purity filtering
    windows_dict = {}
    stage_types = config.analysis.stage_types
    stage_purity = config.analysis.stage_purity

    for stage_type in stage_types:
        label = f"{stage_type}@{int(stage_purity * 100)}%"

        # Define target stages per type
        if stage_type == 'WN':
            target_stages = [0, 1, 2, 3, 4]
        elif stage_type == 'REM':
            target_stages = [4]
        elif stage_type == 'N2N3':
            target_stages = [2, 3]
        else:
            print(f"[WARNING] Sub {row['sub_id']}: Unknown stage type: {stage_type}. Skipping.")
            continue

        # Filter for stage purity
        valid_starts = [
            start_idx for (start_idx, stage_window) in window_candidates
            if np.isin(stage_window, target_stages).mean() >= (stage_purity - 1e-5)
        ]

        # timestamps_print_before = [
        #     sleep_onset_time + timedelta(seconds=int(idx/sfreq))
        #     for idx in valid_starts
        # ]
        # print("BEFORE REMOVING OVERLAP")
        # print(f"Label: {label}")
        # print("Timestamps:", timestamps_print_before)

        # Remove overlapping windows    
        selected_starts = []
        allowed_overlap_sec = config.analysis.allowed_overlap_sec
        allowed_overlap_samples = int(allowed_overlap_sec * sfreq)

        for idx in valid_starts:
            if not any(prev <= idx < prev + window_size_samples - allowed_overlap_samples for prev in selected_starts):
                selected_starts.append(idx)

        if config.run.verbose:
            print(f"{label}: {len(valid_starts)} valid -> {len(selected_starts)} non-overlapping windows")

        timestamps = [
            sleep_onset_time + timedelta(seconds=int(idx/sfreq))
            for idx in selected_starts
        ]
 
        # Save in dictionary
        windows_dict[label] = {
            "count": len(timestamps),
            "timestamps": timestamps
        }
        # print("---------------------------------------------------------------------")
        # print("AFTER REMOVING OVERLAP")
        # print(f"Label: {label}")
        # print("Timestamps:", timestamps)

    # Summary
    if not any(entry["count"] > 0 for entry in windows_dict.values()):
        print(f"[WARNING] Sub {row['sub_id']}: No valid windows found for any stage or purity level.")

    return windows_dict

def selecting_windows_emg(config, sleep_stages, sleep_onset_time, processed_signals):
    None
    return None
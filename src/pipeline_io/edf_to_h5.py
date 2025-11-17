import mne
import re
import traceback
import h5py
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from pathlib import Path
from datetime import datetime, timezone
import numpy as np 
from functools import partial
import warnings
import sys
import os
from collections import Counter


def edf_to_h5(config, rows):
    """Convert all subjects' EDF files to H5, optionally in parallel."""
    failed_subs = []

    if config.run.parallel:
        workers = config.run.num_workers or min(os.cpu_count(), 4)
        print(f"[INFO] Converting {len(rows)} EDF files in parallel with {workers} workers")
        func = partial(_edf_to_h5_one_subject_safe, config)
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(func, rows))

        # Collect failed subjects
        failed_subs = [sub for sub, success in results if not success]

    else:
        print(f"[INFO] Converting {len(rows)} EDF files (not in parallel).")
        for row in rows:
            sub_id, success = _edf_to_h5_one_subject_safe(config, row)
            if not success:
                failed_subs.append(sub_id)

    if failed_subs:
        print(f"[ERROR] The following subjects failed EDF->H5 conversion: {failed_subs}")
    else:
        print("[INFO] All subjects converted successfully.")

    return failed_subs

def _edf_to_h5_one_subject_safe(config, row):
    sub_id = row.get("sub_id", "[unknown]")
    ses_id = row.get("session", "[unknown]")
    psg_id = f"sub-{sub_id}_ses-{ses_id}"

    try:
        edf_to_h5_one_subject(config, row)
        print(f"[SUCCESS] EDF to H5 conversion completed for subject {psg_id}", flush=True)
        return sub_id, True
    except Exception as e:
        print(f"[ERROR] Failed conversion for {psg_id}: {e}", flush=True)
        traceback.print_exc(file=sys.stdout)
        return sub_id, False
    
    

# --- Step 3: Convert EDF to H5 file --- 
def edf_to_h5_one_subject(config, row):
    """Converts a single subject's EDF to H5."""
    # --- Step 3a: Get path from mastersheet --- 
    sub_id = row.get("sub_id", "[unknown]")
    ses_id = row.get("session", "[unknown]")
    psg_id = f"sub-{sub_id}_ses-{ses_id}"
    edf_path = Path(row["edf_path"])
    h5_path = Path(row["h5_path"])
    if h5_path.exists():
        h5_path.unlink()

    # --- Step 3b: Read EDF file ---
    if edf_path.exists():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
    else:
        print(f"    [ERROR] EDF file not found for subject {sub_id}: {edf_path}", flush=True)
        return 

    # --- Step 3c: Get and save metadata ---
    metadata = get_metadata(raw)
    save_metadata_to_h5(metadata, h5_path)
    sfreq_per_channel = get_channel_sampling_freqs(metadata)

    # --- Step 3d: Get and save the EEG signals and their attributes ---
    EEG_TARGETS = {
        # Central
        'C3': ['A2', 'M2'],  # gauche
        'C4': ['A1', 'M1'],  # droite

        # Frontal
        'F3': ['A2', 'M2'],  # gauche
        'F4': ['A1', 'M1'],  # droite

        # Occipital
        'O1': ['A2', 'M2'],  # gauche
        'O2': ['A1', 'M1'],  # droite
    }
    eeg_founded = save_eeg_signals_to_h5(psg_id, raw, sfreq_per_channel, EEG_TARGETS, h5_path, config.run.verbose)
    
    # --- Step 3e: Get and save the other signals and their attributes ---
    SIGNAL_TARGETS = {
        "ECG": ["ECG", "EKG"],
        "EMG_CHIN": ["CHIN", "EMG"],
        "EMG_LEG": ["LEG"],
        "EMG_ARM": ["ARM", "LAT", "RAT"],
        "RESP_AIRFLOW": ["AIRFLOW", "FLOW"] if config.dataset.name.lower() != "hsp_mgb" else ["AIRFLOW"],
        "RESP_ABDOMINAL": ["ABD", "ABDO", "ABDOMINAL"],
        "RESP_THORACIC": ["CHEST", "THORAX", "THORACIC", "THOR"],
        "SP02": ["SAO2", "SPO2"],
        "POSITION": ["POS", "POSITION"],
        "HR_DERIVATED": ["DHR"],
        "HR": ["HR"],
        "EOG": ["EOG","E1", "E2","ROC","LOC"] # at the end on purpose
        }
    extract_and_save_channels(psg_id, raw, sfreq_per_channel, SIGNAL_TARGETS, eeg_founded,h5_path, config.run.verbose)



# --- Step 3c: Get and save metadata ---
def get_metadata(raw):
    """Extract consistent metadata from an MNE Raw object."""
    start_time = raw.info['meas_date']
    start_time = datetime.fromisoformat(str(start_time)).replace(tzinfo=timezone.utc)
    sfreq_global = raw.info['sfreq']
    duration_samples = raw.n_times
    duration_sec = duration_samples / sfreq_global
    metadata = {"start_time": np.string_(start_time.isoformat()),
                "sfreq_global": sfreq_global,
                "duration_samples": duration_samples,
                "duration_sec": duration_sec,
                "channel_names": raw.info['ch_names'],
                "sfreq_per_channel": raw._raw_extras[0]['n_samps']
                }
    return metadata

def save_metadata_to_h5(metadata, h5_path):
    """Save metadata dictionary to an HDF5 file under group 'metadata'."""
    with h5py.File(h5_path, "a") as f:  # Use "a" to append if file exists
        metadata_grp = f.require_group("metadata")
        for key, value in metadata.items():
            metadata_grp.attrs[key] = value

def get_channel_sampling_freqs(metadata):
    """
    Return {channel_name: sampling_frequency} using MNE's public API.
    """
    return {
        ch : fs 
        for ch, fs in zip(metadata["channel_names"],metadata["sfreq_per_channel"])
    }

# --- Step 3d: Get and save the EEG signals and their attributes ---
def extract_eeg_channel(raw, sfreq_per_channel, target, refs, sub_id):
    channel_names = raw.ch_names.copy() 
    harmon_to_raw = {ch: ch for ch in channel_names} 

    # Adjust for SHHS
    if "shhs" in sub_id.lower():
        eeg_channels = [ch for ch in channel_names if "EEG" in ch.upper()]
        if eeg_channels:
            ch_with_more = max(eeg_channels, key=len)
            ch_with_less = min(eeg_channels, key=len)
            rename_map = {ch_with_more: "C3-M2", ch_with_less: "C4-M1"}
            for orig, harmon in rename_map.items():
                if orig in channel_names:
                    harmon_to_raw[harmon] = orig

    # Adjust for MESA
    if "mesa" in sub_id.lower():
        required_channels = ["EEG1", "EEG2", "EEG3"]
        missing = [ch for ch in required_channels if ch not in channel_names]
        if missing:
            print(f"    [ERROR] Sub {sub_id} has EEG issues: {channel_names}", flush=True)
        else:
            rename_map = {"EEG1": "Fz-Cz", "EEG2": "Cz-Oz", "EEG3": "C4-M1"}
            for orig, harmon in rename_map.items():
                if orig in channel_names:
                    harmon_to_raw[harmon] = orig

    # Now loop over harmonized channels
    harmonized_name = f"EEG_{target}"

    # Case 1: Bipolar channel like "C3-A2"
    for harmon_name, raw_name in harmon_to_raw.items():
        if harmon_name.startswith(target + "-") or raw_name.startswith(target + "-"):
            signal = raw.get_data(picks=raw_name)[0]
            fs = sfreq_per_channel.get(raw_name)
            if fs is None:
                print(f"[WARNING] No sampling rate found for channel '{raw_name}' for subject {sub_id}", flush=True)
                return None, {
                    'harmonized_name': harmonized_name,
                    'raw_names': [raw_name],
                    'fs': None,
                    'type': 'EEG',
                }
            return signal, {
                'harmonized_name': harmonized_name,
                'raw_names': [raw_name],
                'fs': fs,
                'type': 'EEG',
            }

    # Case 2: Separate channels (C3 and A2)
    for ref in refs:
        if target in channel_names and ref in channel_names:
            sig1 = raw.get_data(picks=target)[0]
            sig2 = raw.get_data(picks=ref)[0]
            min_len = min(len(sig1), len(sig2))
            signal = sig1[:min_len] - sig2[:min_len]
            fs = sfreq_per_channel.get(target)
            if fs is None:
                print(f"[WARNING] No sampling rate found for channel '{target}' for subject {sub_id}", flush=True)
                return None, {
                    'harmonized_name': harmonized_name,
                    'raw_names': [target, ref],
                    'fs': None,
                    'type': 'EEG',
                }
            return signal, {
                'harmonized_name': harmonized_name,
                'raw_names': [target, ref],
                'fs': fs,
                'type': 'EEG',
            }

    # Case 3: Not found
    return None, {
        'harmonized_name': harmonized_name,
        'raw_names': [],
        'fs': None,
        'type': 'EEG',
    }

def save_eeg_signals_to_h5(sub_id, raw, sfreq_per_channel, targets, h5_path, verbose):
    extracted_eeg_count = 0
    eeg_founded = []

    with h5py.File(h5_path, "a") as f:
        for target, refs in targets.items():
            signal, attrs = extract_eeg_channel(raw, sfreq_per_channel, target, refs, sub_id)

            if signal is not None:
                eeg_founded.append(attrs['raw_names'])
                extracted_eeg_count += 1
                path = f"signals/EEG/{attrs['harmonized_name']}"
                dset = save_signal_to_h5(f, path, signal, attrs)
                if dset is not None and verbose:
                    print(f"[OK] {sub_id} {path} : {attrs['raw_names']}")
    
    # Flatten list of lists and remove duplicates
    eeg_founded = [ch for sublist in eeg_founded for ch in sublist]
    eeg_founded = list(set(eeg_founded))

    return eeg_founded
                
# --- Step 3e: Get and save the other signals and their attributes ---
def find_channels(channel_names, signal_name, keywords, sub_id):

    # Find channels matching any keyword (case-insensitive) 
    matches = [
        ch for ch in channel_names
        if any(
            k.upper() in re.sub(r'(?<=.)\.(?=.)', '', ch).upper()
            for k in keywords
        )
    ]
        
    # Remove duplicates added by MNE (keep only the first occurrence, -0)
    if any(re.search(r'-0$', ch) for ch in matches):
        matches = [ch for ch in matches if not re.search(r'-\d+$', ch) or ch.endswith('-0')]
    
    # Remove duplicated that have Off in their names
    matches = [ch for ch in matches if "OFF" not in ch.upper()]

    # Select the ECG if more than two
    if len(matches) > 2 and any('ECG' in m.upper() for m in matches):
        
        exact_ecg = [m for m in matches if m.upper() == "ECG"]
        exact_ekg = [m for m in matches if m.upper() == "EKG"]

        if exact_ecg:
            selected = exact_ecg
        elif exact_ekg:
            selected = exact_ekg
        else:
            selected = [m for m in matches if any(tag in m.upper() for tag in ["ECG-RA", "ECG-LL"])] 
            if len(selected) != 2:
                selected = [m for m in matches if any(tag in m.upper() for tag in ["ECG-LA", "ECG-RA"])] 
            if len(selected) != 2:
                selected = [m for m in matches if "ECG-LL" in m.upper()]
            if not selected:
                selected = matches[:2]
        matches = selected

    # Handle leg biploar channels (take the positives)
    if len(matches) >= 4 and sum("+" in ch for ch in matches) >= 2 and sum("-" in ch for ch in matches) >= 2:
        matches = [ch for ch in matches if "+" in ch]

    # Handle when monopolar and differential channels (keep monopolars only)
    if any("-" in ch for ch in matches) and  len(matches) >= 3:
        matches = [ch for ch in matches if "-" in ch]
        if len(matches)>1:
            for ch in matches:
                left, right = ch.split("-", 1)
                prefix_left = ''.join([c for c in left if c.isalpha()])
                prefix_right = ''.join([c for c in right if c.isalpha()])
                print(prefix_left, "and", prefix_right)
                if prefix_left == prefix_right:
                    print(prefix_left, "==", prefix_right)
                    matches = [ch]
                elif prefix_right == "REF":
                    matches = matches[:2]
                    

    # If both EMG and CHIN remove EMG form matches
    if any("EMG" in ch.upper() for ch in matches) and any("CHIN" in ch.upper() for ch in matches):
        matches = [ch for ch in matches if "EMG" in ch.upper()]

    # Extremly specific -> decide to keep Arm1 and Arm2
    if len(matches) > 2 and any("ARM1" in m.upper() for m in matches) and any("ARM2" in m.upper() for m in matches):
        matches = [m for m in matches if "ARM1" in m.upper() or "ARM2" in m.upper()]

    if len(matches) == 0:   
        print(f"    No {signal_name} channels found for subject {sub_id}")
        return []
    elif len(matches) == 1:
        return matches  # single channel list
    
    # len(matches) >= 2
    # Remove the matched keyword part to avoid confusing
    cleaned_matches = []
    for ch in matches:
        ch_upper = ch.upper()
        for k in keywords:
            ch_upper = ch_upper.replace(k.upper(), "") 
        cleaned_matches.append(ch_upper)

    if len(set(cleaned_matches)) == 1:
        common_string = cleaned_matches[0]
        cleaned_matches = [ch.replace(common_string, '') for ch in matches]

    # Overwrite clean match to adjust cases like [CHIN, CHIN3]
    if ('' in cleaned_matches) and any(ch in {'3', '.'} for ch in cleaned_matches):
        print("Dot OR 3 before:", cleaned_matches)
        cleaned_matches = ['2' if ch in {'3', '.'} else ch for ch in cleaned_matches]
        print("Dot OR 3 after :", cleaned_matches)

    # Handle bad suffix
    for suffix in ['-REF', '-E1']:
        count = sum(m.upper().endswith(suffix) for m in matches)
        if count >= 2:
            cleaned_matches = [
                m[:-len(suffix)] if m.upper().endswith(suffix) else m
                for m in matches
            ]

    # Identify left/right using cleaned channel names
    left_candidates = [matches[i] for i, ch in enumerate(cleaned_matches) if "L" in ch or "1" in ch or "OLD" in ch or "UP" in ch]
    right_candidates = [matches[i] for i, ch in enumerate(cleaned_matches) if "R" in ch or "2" in ch or "NEW" in ch or "DOWN" in ch]

    # Handle when clean match removed full name
    if not left_candidates and not right_candidates:
        left_candidates = [ch for ch in matches if "L" in ch or "1" in ch or "1" in ch]
        right_candidates = [ch for ch in matches if "R" in ch or "2" in ch]
    elif not left_candidates:
        left_candidates = [ch for ch in matches if ch not in right_candidates]
    elif not right_candidates:
        right_candidates = [ch for ch in matches if ch not in left_candidates]

    if len(left_candidates) == 1 and len(right_candidates) == 1:
        return [left_candidates[0], right_candidates[0]]
    else:
        print(f"    [ERROR] Cannot uniquely identify left/right in {signal_name} channels {matches} for subject {sub_id}")
        return []
    

def extract_and_save_channels(sub_id, raw, sfreq_per_channel, SIGNAL_TARGETS, eeg_founded, h5_path, verbose):
    with h5py.File(h5_path, "a") as f:
        channels = list(raw.ch_names)
        remaining_channels = [ch for ch in channels if ch not in eeg_founded]
        remaining_channels = [
            ch for ch in channels
            if not (
                ch.startswith("O1-") or 
                ch.startswith("O2-") 
            )
        ]
        for signal_name, keywords in SIGNAL_TARGETS.items():
            chs = find_channels(remaining_channels, signal_name, keywords, sub_id)
            for ch in chs:
                if ch in remaining_channels:
                    remaining_channels.remove(ch) 
            
            type_name = signal_name.split("_")[0].upper() 

            if len(chs) == 0 :
                continue
            
            # Extract signal data
            if len(chs) == 1:
                ch = chs[0]
                signal = raw.get_data(picks=ch)[0]
                fs = sfreq_per_channel.get(ch)
                raw_names = [ch]

                # Compose path in HDF5
                path = f"signals/{type_name.upper()}/{signal_name.upper()}"
                attrs = {
                    "raw_names": [str(rn) for rn in raw_names],
                    "fs": fs if fs is not None else "None",
                    "type": type_name.upper(),  # e.g. 'ECG', 'EOG', 'EMG'
                    }
                dset = save_signal_to_h5(f, path, signal, attrs)
                if dset is not None and verbose:
                    print(f"[OK] {sub_id} {path} : {attrs['raw_names']}")
                    
            else:
                ch_L, ch_R = chs
                signal_L = raw.get_data(picks=ch_L)[0]
                signal_R = raw.get_data(picks=ch_R)[0]
                fs = sfreq_per_channel.get(ch_L)

                # Save Left Channel
                harmonized_ch_L = f"{signal_name.upper()}_L"
                path_L = f"signals/{type_name}/{harmonized_ch_L}"
                attrs_L = {"raw_names": [ch_L], "fs": fs if fs else "None", "type": type_name}
                dset = save_signal_to_h5(f, path_L, signal_L, attrs_L)
                if dset is not None and verbose:
                    print(f"[OK] {sub_id} {path_L} : {attrs_L['raw_names']}")

                # Save Right Channel
                harmonized_ch_R = f"{signal_name.upper()}_R"
                path_R = f"signals/{type_name}/{harmonized_ch_R}"
                attrs_R = {"raw_names": [ch_R], "fs": fs if fs else "None", "type": type_name}
                dset = save_signal_to_h5(f, path_R, signal_R, attrs_R)
                if dset is not None and verbose:
                    print(f"[OK] {sub_id} {path_R} : {attrs_R['raw_names']}")

                # Save Differential Channel (L - R) if required
                if (ch_R.upper() == "ECG-RA") and (len(signal_L) == len(signal_R)):
                    signal_diff = signal_L - signal_R
                    path = f"signals/{type_name.upper()}/{signal_name.upper()}"
                    attrs = {
                        "raw_names":  [str(ch) for ch in chs],
                        "fs": fs if fs is not None else "None",
                        "type": type_name,  
                        }
                    dset = save_signal_to_h5(f, path, signal_diff, attrs)
                    if dset is not None and verbose:
                        print(f"[OK] {sub_id} {path} : {attrs['raw_names']}")

def save_signal_to_h5(f, path, signal, attrs):
    dset = f.create_dataset(path, data=signal.astype("float32"))
    for k, v in attrs.items():
        if k == 'harmonized_name':
            continue  # Do not store this field
        dset.attrs[k] = [str(i) for i in v] if isinstance(v, list) else (v if v is not None else "None")
    return dset
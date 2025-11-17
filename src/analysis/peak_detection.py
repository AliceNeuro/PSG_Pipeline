import os
from pathlib import Path
import scipy.io as sio
import subprocess
import numpy as np
from itertools import groupby
import neurokit2 as nk
import time

def peak_detection(config, ecg_data, tmp_dir_sub):
    """
    Detect R-peaks in ECG data using the specified method and fallback options.

    Parameters
    ----------
    ecg_data {dict} Dictionary containing ECG signals and metadata.
    tmp_dir_sub : pathlib.Path
        Temporary directory for storing intermediate files.
    rpeaks_detection_method : str
        Primary method for R-peak detection ("Farhad_Kenn" or "neurokit").

    Returns
    -------
    clean_rpeaks : np.ndarray or None
        Array of cleaned R-peaks indices or None if no valid R-peaks are found.
    chosen_signal : str or None
        Key of the signal used for R-peak detection or None if no signal is valid.
    """
    sub_id = tmp_dir_sub.name
    ecg_file = tmp_dir_sub / f"ecg_{sub_id}.mat"

    # Priority order
    rpeaks_detection_method = config.analysis.rpeaks_detection_method
    if rpeaks_detection_method == "Farhad_Kenn":
        method_order = [
            ("ECG", rpeaks_detection_method),
            ("ECG_L", rpeaks_detection_method),
            ("ECG_R", rpeaks_detection_method),
        ]
    else:
        method_order = [
            ("ECG", "neurokit"),
            ("ECG_L", "neurokit"),
            ("ECG_R", "neurokit"),
        ]

    clean_rpeaks = None
    clean_rpeaks_mask = None
    chosen_signal = None

    # Track best candidate if no method reaches "strict" threshold
    best_candidate = {
        "percentage": -1,
        "rpeaks": None,
        "mask": None,
        "signal": None
    }

    for signal_key, method in method_order:
        # Check that the signal exists in ecg_data
        if ecg_data is not None and signal_key in ecg_data and ecg_data[signal_key] is not None:
            if config.run.verbose:
                print(f"Peak Detection for {signal_key} with {method}")

            ecg = ecg_data[signal_key].get("signal")
            sfreq_ecg = ecg_data[signal_key].get("sfreq_signal")
            
            # Skip if signal or sfreq is missing
            if ecg is None or sfreq_ecg is None:
                continue

            rpeaks, rpeaks_artifact_mask = get_rpeaks(ecg, sfreq_ecg, method, ecg_file)

            # Check if detected rpeaks are valid
            if len(rpeaks) > 10 and not np.all(np.isnan(rpeaks)) and not np.all(rpeaks <= 0):
                mask = get_clean_rpeaks_mask(config, len(ecg), sfreq_ecg, rpeaks, rpeaks_artifact_mask)
                rpeaks_clean = rpeaks[mask[rpeaks]]
                percentage_clean = (len(rpeaks_clean) / len(rpeaks)) * 100

                if config.run.verbose:
                    print("Percentage of clean Rpeaks:", percentage_clean, "%")

                # Save best candidate so far
                if percentage_clean > best_candidate["percentage"]:
                    best_candidate = {
                        "percentage": percentage_clean,
                        "rpeaks": rpeaks_clean,
                        "mask": mask,
                        "signal": signal_key
                    }

                # Strict acceptance threshold
                if percentage_clean > 80 and not np.all(np.isnan(rpeaks_clean)) and not np.all(rpeaks_clean <= 0):
                    chosen_signal = signal_key
                    clean_rpeaks = rpeaks_clean
                    clean_rpeaks_mask = mask
                    break
      

    # If nothing broke early, fall back to best candidate
    if clean_rpeaks is None:
        if best_candidate["percentage"] > 10:  # fallback acceptance threshold
            clean_rpeaks = best_candidate["rpeaks"]
            clean_rpeaks_mask = best_candidate["mask"]
            chosen_signal = best_candidate["signal"]
            if config.run.verbose:
                print(f"[INFO] {sub_id}: Falling back to best candidate "
                      f"{chosen_signal} ({best_candidate['percentage']:.1f}% clean)")
        else:
            print(f"[INFO] {sub_id} No valid R-peaks found in any ECG channels ({best_candidate['percentage']}% clean).")
            return None, None, None

    return clean_rpeaks, clean_rpeaks_mask, chosen_signal


def get_rpeaks(ecg, sfreq_ecg, method, ecg_file):
    """
    Extract R-peaks using the specified method.

    Parameters
    ----------
    ecg : np.ndarray
        ECG signal array.
    sfreq_ecg : float
        Sampling frequency of the ECG signal.
    method : str
        Method for R-peak detection ("Farhad_Kenn" or "neurokit").
    ecg_file : pathlib.Path
        Path to save intermediate MATLAB files.

    Returns
    -------
    rpeaks : np.ndarray
        Array of detected R-peak indices.
    rpeaks_artifact_mask : np.ndarray
        Boolean mask indicating good (True) or artifact (False) R-peaks.
    """
    
    if method == "Farhad_Kenn":
        detect_rpeaks_farhad_kenn(ecg, sfreq_ecg, ecg_file, verbose = True)
        rpeaks_mat = sio.loadmat(ecg_file)
        rpeaks = rpeaks_mat['R_Index'].flatten().astype(int) - 1
        rpeaks_artifact_mask = np.ones_like(rpeaks, dtype=bool)
        rpeaks_artifact_mask[rpeaks_mat['Art_Index'].flatten().astype(int) - 1] = False
    else:
        rpeaks = nk.ecg_peaks(ecg, sampling_rate=sfreq_ecg)[1]['ECG_R_Peaks']
        rpeaks_artifact_mask = np.ones_like(rpeaks, dtype=bool)  # All good
    
    return rpeaks, rpeaks_artifact_mask


def detect_rpeaks_farhad_kenn(ecg, sfreq_ecg, ecg_file, verbose = True, max_attempts=5, wait_sec=10):
    """
    Run MATLAB script to detect R-peaks using Farhad Kenn's method.

    Parameters
    ----------
    ecg : np.ndarray
        ECG signal array.
    sfreq_ecg : float
        Sampling frequency of the ECG signal.
    ecg_file : pathlib.Path
        Path to save intermediate MATLAB files.
    verbose : bool
        Whether to print MATLAB command output.
    """
    # Link to matlab file
    project_src_root = Path(__file__).resolve().parents[1] 
    mat_rpeaks= project_src_root / "external_tools/matlab/mros_rpeaks_detection/get_Farhad_Kenn_rpeaks.m"

    # Save ECG signal and sampling frequency to the MATLAB file
    sio.savemat(ecg_file, {'ecg': ecg, 'fs': sfreq_ecg})
            
    # Define the MATLAB command to run the R-peak detection script
    command = [
        "/usr/local/matlab/R2024a/bin/matlab", # "/usr/local/matlab/R2024a/bin/matlab", # "/Applications/MATLAB_R2025a.app/bin/matlab"
        "-nojvm",           # No Java, lighter, no GUI
        "-nosplash",        # No splash screen
        "-nodesktop",       # No desktop GUI
        "-softwareopengl",  # Safe for headless servers
        "-batch",           # Run command non-interactively 
        f'addpath("{Path(mat_rpeaks).parent}"); get_Farhad_Kenn_rpeaks("{ecg_file}")'
    ]

        # Retry loop
    for attempt in range(1, max_attempts + 1):
        try:
            subprocess.run(
                command,
                check=True,
                stdout=None if verbose else subprocess.DEVNULL,
                stderr=None if verbose else subprocess.DEVNULL
            )
            # Success: exit the loop
            return
        except subprocess.CalledProcessError as e:
            err_msg = str(e)
            if "License Manager Error -16" in err_msg:
                if attempt < max_attempts:
                    print(f"[MATLAB] License busy, retry {attempt}/{max_attempts} in {wait_sec}s...")
                    time.sleep(wait_sec)
                else:
                    raise RuntimeError(f"MATLAB license unavailable after {max_attempts} attempts.")
            else:
                # Any other MATLAB error: raise immediately
                raise RuntimeError(f"MATLAB failed with error: {e}")

def get_clean_rpeaks_mask(config, signal_length, sfreq_ecg, rpeaks, rpeak_artifact_mask):
    """
    Generate a boolean mask indicating high-quality regions of an ECG signal.

    Parameters
    ----------
    signal_length : int
        Total number of samples in the ECG signal.
    rpeaks : np.ndarray
        Array of R-peak indices.
    rpeak_artifact_mask : array-like
        Boolean array indicating good (True) or artifact (False) status for each R-peak.
    fs : float
        Sampling frequency of the ECG signal.

    Returns
    -------
    clean_ecg_mask : np.ndarray (dtype=bool)
        Boolean array of length `signal_length`, where True indicates good-quality ECG segments.
    """

    # Copy the mask to avoid modifying the input
    rpeak_mask = np.array(rpeak_artifact_mask, dtype=bool)

    # Compute RR intervals in seconds
    rr_intervals = np.diff(rpeaks) / sfreq_ecg
    q1, q3 = np.percentile(rr_intervals, (25, 75))
    iqr = q3 - q1

    # Define physiological bounds 
    rr_min = max(q1 - 4 * iqr, config.analysis.min_rri_sec)   # 200 bpm = 0.3s
    rr_max = min(q3 + 4 * iqr, config.analysis.max_rri_sec)   # 20 bpm = 3.0s 

    # Flag outliers as bad (and next beat too)
    outlier_ids = np.where((rr_intervals < rr_min) | (rr_intervals > rr_max))[0]
    rpeak_mask[outlier_ids] = False
    if outlier_ids.size > 0:
        next_beats = outlier_ids + 1
        next_beats = next_beats[next_beats < len(rpeak_mask)] 
        rpeak_mask[next_beats] = False

    # Initialize signal-quality mask
    pad_samples = int(round(0.2 * sfreq_ecg))  # 200 ms padding
    clean_mask = np.zeros(signal_length, dtype=bool)

    idx = 0
    for is_good, group in groupby(rpeak_mask):
        count = len(list(group))
        if is_good:
            start = max(rpeaks[idx] - pad_samples, 0)
            end = min(rpeaks[idx + count - 1] + pad_samples + 1, signal_length)
            clean_mask[start:end] = True
        idx += count

    # Remove short good segments (<10 seconds)
    min_good_len = int(10 * sfreq_ecg)
    clean_rpeaks_mask = clean_mask.copy()
    idx = 0
    for is_good, group in groupby(clean_mask):
        count = len(list(group))
        if is_good and count <= min_good_len:
            clean_rpeaks_mask[idx:idx + count] = False
        idx += count

    return clean_rpeaks_mask


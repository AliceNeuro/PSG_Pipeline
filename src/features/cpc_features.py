import subprocess
from fsspec import register_implementation
import hrvanalysis
import wfdb
import os
import mne
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import interp1d
from scipy.stats import linregress
from nitime.analysis import MTCoherenceAnalyzer
from nitime.timeseries import TimeSeries
from nitime.algorithms import mtm_cross_spectrum
from datetime import timedelta
import platform

system_name = platform.system() 

def extract_cpc(config, tmp_dir_sub, sleep_onset_time, ecg_data, windows_dict_ecg):
    if ecg_data is None or windows_dict_ecg is None:
        return None
    
    feat_names_cpc = [
        'log_pwr_VLFC', 'log_pwr_LFC', 'log_pwr_HFC', 
        'log_pwr_VL2LH', 'log_pwr_H2L']
    
    ecg_signal = ecg_data["signal"]
    rpeaks = ecg_data["clean_rpeaks"]
    sfreq_ecg = ecg_data["sfreq_signal"]
    
    # --- Step 2: Get RRI and EDR alligned and resampled ---
    target_sfreq = 2.0
    time_resampled, rri_resampled, edr_resampled = resample_rri_edr(ecg_signal, rpeaks, sfreq_ecg, target_sfreq, config, tmp_dir_sub)

    # If resampling failed or produced empty timeline, return NaNs per stage
    if time_resampled is None or time_resampled.size == 0:
        features_by_stage = []
        for stage_key, info in windows_dict_ecg.items():
            stage_type = stage_key.split('@')[0]
            features_by_stage.append({**{k: np.nan for k in feat_names_cpc}, 'stage_type': stage_type})
        return features_by_stage
    
    psg_id = Path(tmp_dir_sub).name
    window_size_sec = config.analysis.window_size_min * 60
    window_size_samples = int(window_size_sec * target_sfreq)

    # Compute window start indices relative to first resampled time
    if time_resampled[0] > 120:
        print(f"[INFO] {psg_id} shift (first clean rpeak): {time_resampled[0]:.2f}sec")
    trim_start_time = sleep_onset_time + timedelta(seconds=time_resampled[0])
    #trim_start_time = sleep_onset_time 

    features_by_stage = []

    for stage_key, info in windows_dict_ecg.items():
        stage_type = stage_key.split('@')[0]
        timestamps = info.get('timestamps', [])

        # If no windows, return NaN for all features
        if not timestamps:
            features_by_stage.append({**{k: np.nan for k in feat_names_cpc}, 'stage_type': stage_type})
            continue
        
        window_start_ids = [
            int(round((timestamp - trim_start_time).total_seconds() * target_sfreq))
            for timestamp in timestamps
        ]

        # Initialize storage for features
        feats_all_windows = {k: [] for k in feat_names_cpc}

        for start in window_start_ids:
            end = start + window_size_samples

            if start < 0 or end > len(time_resampled):
                continue

            rri_seg = rri_resampled[start:end]
            edr_seg = edr_resampled[start:end]
            rri_seg = nandetrend(rri_seg)
            edr_seg = nandetrend(edr_seg)

            if np.isnan(rri_seg).any() or np.isnan(edr_seg).any():
                print(f"[WARNING] Sub {psg_id}: rri_seg or edr_seg contain NaN values.")
                continue

            cpc_features = get_cpc_features_window(rri_seg, edr_seg, target_sfreq)
            for k in feat_names_cpc:
                feats_all_windows[k].append(cpc_features.get(k, np.nan))

        # If all windows were skipped, fill with NaN
        if not any(len(v) > 0 for v in feats_all_windows.values()):
            features_by_stage.append({**{k: np.nan for k in feat_names_cpc}, 'stage_type': stage_type})
            continue

        # Take mean across windows
        feat_stage = {k: np.nanmean(v) for k, v in feats_all_windows.items()}
        feat_stage['stage_type'] = stage_type
        features_by_stage.append(feat_stage)

    return features_by_stage

def get_edr_from_ecg(ecg_signal, rpeaks, sfreq_ecg, tmp_dir_sub):
    """
    Runs external EDR tool and returns (sample_indices, edr_values).
    Does not interpolate to ECG timeline.
    """
    # Set up paths
    project_src_root = Path(__file__).resolve().parents[1] 
    if system_name == "Linux":
        edr_module = project_src_root / "external_tools/c_modules/edr_linux"
    elif system_name == "Darwin":  # macOS
        edr_module = project_src_root / "external_tools/c_modules/edr_macOS"

    tmp_dir_sub = Path(tmp_dir_sub)
    sub_id = tmp_dir_sub.name
    edr_file = tmp_dir_sub / f"edr_{sub_id}_output.txt"
    record_name = "rec"

    # Write temporary ECG signal and annotations
    wfdb.wrsamp(
        record_name,
        fs=sfreq_ecg,
        units=['uV'],
        sig_name=['ECG'],
        p_signal=ecg_signal.reshape(-1, 1),
        write_dir=str(tmp_dir_sub),
    )

    # Write R-peak annotations 
    wfdb.wrann(
        record_name,
        extension='atr',
        sample=rpeaks,
        label_store=np.ones(len(rpeaks), dtype=int),
        fs=sfreq_ecg,
        write_dir=str(tmp_dir_sub),
    )

    # --- Fix: set LD_LIBRARY_PATH for subprocess ---
    env = os.environ.copy()
    # wfdb_lib_dir = str(project_src_root / "external_tools/wfdb/lib")
    wfdb_lib_dir = "/wynton/home/leng/alice-albrecht/projects/wfdb/lib"
    env["LD_LIBRARY_PATH"] = env.get("LD_LIBRARY_PATH", "") + f":{wfdb_lib_dir}"

    # Run external EDR binary
    with open(edr_file, "w") as f:
        subprocess.check_call(
            [str(edr_module), "-r", record_name, "-i", "atr", "-v"],
            cwd=tmp_dir_sub,
            stdout=f,
            env=env,   
        )
        
    # Load output 
    df_edr = pd.read_csv(edr_file, header=None,  sep=r'\s+')
    if df_edr.shape[1] < 2 or df_edr.shape[0] < 2: # Not enough data produced
        return np.array([], dtype=float), np.array([], dtype=float)
    
    # exclude first index and value to align with RRI assigned to ending R-peak 
    edr_indices = df_edr[0].values[:-1]
    edr_values = df_edr[1].values[:-1] 

    return edr_indices, edr_values


def resample_rri_edr(ecg_signal, rpeaks, sfreq_ecg, target_sfreq, config, tmp_dir_sub):
    # --- Step 1: Compute RRI ---
    rri_values = np.diff(rpeaks)
    rri_indices = rpeaks[:-1] # assign RRI to the second R-peak

    # --- Step 2: Filter RRI based on config ---
    #rri_values = np.array(hrvanalysis.get_nn_intervals(rri_values/sfreq_ecg*1000, verbose=False))/1000*sfreq_ecg
    # here is the code to avoid get_nn_intervals just remove min/max
    min_rri_samples = config.analysis.min_rri_sec * sfreq_ecg
    max_rri_samples = config.analysis.max_rri_sec * sfreq_ecg
    valid_mask = (rri_values >= min_rri_samples) & (rri_values <= max_rri_samples)
    rri_values_valid = rri_values[valid_mask]
    rri_indices_valid = rri_indices[valid_mask]
    if rri_values_valid.size == 0 or rri_indices_valid.size == 0:
        return np.array([]), np.array([]), np.array([])
  
    # --- Step 3: Extract EDR from ECG ---
    # edr_indice is also in smaples exactly equal to rpeaks
    edr_indices, edr_values = get_edr_from_ecg(ecg_signal,
                                            rpeaks, 
                                            sfreq_ecg, 
                                            tmp_dir_sub)
    if edr_indices.size == 0 or edr_values.size == 0:
        return np.array([]), np.array([]), np.array([])

    #  # --- Step 4: Crop to overlapping region ---
    start_sec = max(rri_indices_valid.min(), edr_indices.min()) / sfreq_ecg
    stop_sec  = min(rri_indices_valid.max(), edr_indices.max()) / sfreq_ecg
    if not np.isfinite(start_sec) or not np.isfinite(stop_sec) or stop_sec <= start_sec:
        return np.array([]), np.array([]), np.array([])

    rri_mask_crop = (rri_indices_valid / sfreq_ecg >= start_sec) & (rri_indices_valid / sfreq_ecg <= stop_sec)
    edr_mask_crop = (edr_indices / sfreq_ecg >= start_sec) & (edr_indices / sfreq_ecg <= stop_sec)

    rri_indices_cropped = rri_indices_valid[rri_mask_crop]
    rri_values_cropped = rri_values_valid[rri_mask_crop]
    edr_indices_cropped = edr_indices[edr_mask_crop]
    edr_values_cropped = edr_values[edr_mask_crop]

    if rri_indices_cropped.size < 2 or edr_indices_cropped.size < 2:
        # Need at least 2 points to interpolate meaningfully
        return np.array([]), np.array([]), np.array([])

    # --- Step 5: Interpolate to 2Hz timeline ---
    # Convert to seconds
    rri_times_sec = rri_indices_cropped / sfreq_ecg
    rri_sec = rri_values_cropped / sfreq_ecg 
    edr_times_sec = edr_indices_cropped / sfreq_ecg
    edr_vals = edr_values_cropped 

    # Create 2Hz timeline
    dt = 1.0 / target_sfreq
    time_resampled = np.arange(start_sec, stop_sec, dt)
    if time_resampled.size == 0:
        return np.array([]), np.array([]), np.array([])

    # Interpolate using cubic splines
    rri_interp = interp1d(rri_times_sec, rri_sec, kind='linear', bounds_error=False,fill_value=np.nan)
    edr_interp = interp1d(edr_times_sec, edr_vals, kind='linear', bounds_error=False, fill_value=np.nan)
    rri_resampled = rri_interp(time_resampled)
    edr_resampled = edr_interp(time_resampled)

    return time_resampled, rri_resampled, edr_resampled


def get_cpc_features_window(rri_seg, edr_seg, target_sfreq,
                     freq_vl=[0, 0.01], freq_l=[0.01, 0.1], freq_h=[0.1, 0.4],
                     use_max_two=True): 
    try:
        # Compute cross-spectral coherence
        ts = TimeSeries(np.array([rri_seg, edr_seg]), sampling_rate=target_sfreq)
        analyzer = MTCoherenceAnalyzer(ts)

        freqs = analyzer.frequencies  # frequency bins
        coh = analyzer.coherence[0, 1]  # coherence between RRI and EDR

        # Cross spectral density (CSD)
        # CPC is coherence × |CSD|^2, using multitaper method
        csd = mtm_cross_spectrum(
            analyzer.spectra[0], 
            analyzer.spectra[1],
            (analyzer.weights[0], analyzer.weights[1]),
            sides='onesided')
        csd = np.abs(csd) ** 2

        cpc = coh * csd  # final CPC values across frequency bins

        # Compute band powers
        vlfc = band_power(cpc, freqs, freq_vl, use_max_two)
        lfc = band_power(cpc, freqs, freq_l, use_max_two)
        hfc = band_power(cpc, freqs, freq_h, use_max_two)

        # Log and ratio
        log_vlfc = safe_log(vlfc)
        log_lfc = safe_log(lfc)
        log_hfc = safe_log(hfc)
        log_vl2lh = log_vlfc - safe_log(lfc + hfc) if not np.isnan(log_vlfc) else np.nan
        log_h2l = log_hfc - log_lfc if not np.isnan(log_hfc) and not np.isnan(log_lfc) else np.nan

        cpc_features = {
            "log_pwr_VLFC": log_vlfc,
            "log_pwr_LFC": log_lfc,
            "log_pwr_HFC": log_hfc,
            "log_pwr_VL2LH": log_vl2lh,
            "log_pwr_H2L": log_h2l,
        }
    except Exception as e:
        # In case nitime fails (e.g., short segment), return NaNs
        cpc_features = {
            "log_pwr_VLFC": np.nan,
            "log_pwr_LFC": np.nan,
            "log_pwr_HFC": np.nan,
            "log_pwr_VL2LH": np.nan,
            "log_pwr_H2L": np.nan,
        }

    return cpc_features


# Other utils functions
def nandetrend(x):
    t = np.arange(len(x))
    ids = ~np.isnan(x)
    if np.sum(ids) < 2:  # not enough points to fit a line
        return x  # return as is
    res = linregress(t[ids], x[ids])
    x_detrended = x - (t*res.slope + res.intercept)
    return x_detrended

def band_power(cpc_vals, f, band, use_max_two):
    cpc_vals = np.asarray(cpc_vals)
    f = np.asarray(f)
    mask = (f >= band[0]) & (f < band[1])
    vals_in_band = cpc_vals[mask]
    vals_in_band = vals_in_band[~np.isnan(vals_in_band)]

    if vals_in_band.size == 0:
        return 0.0

    if use_max_two:
        if vals_in_band.size == 1:
            return float(vals_in_band[0])
        else:
            top_vals = np.sort(vals_in_band)[-2:]
            return float(np.sum(top_vals))
    else:
        return float(np.sum(vals_in_band))
    
def safe_log(x):
    # Log-transformed powers and ratios
    return float(np.log(x)) if x > 0 else np.nan


# def align_edr_to_ecg_timeline(edr_sample_indices, edr_values, ecg_len):
#     """
#     Interpolates EDR values onto the ECG sample timeline.
#     """

#     ecg_samples = np.arange(ecg_len)
#     print("Lengths for RRI:", len(ecg_samples), len(edr_sample_indices))

#     interp_func = interp1d(
#         edr_sample_indices, edr_values,
#         kind="linear",
#         bounds_error=False,
#         fill_value=np.nan  
#     )
#     return interp_func(ecg_samples)

# def align_rri_to_ecg_timeline(rpeaks,
#                                ecg_len,
#                                sfreq_ecg,
#                                min_rri_sec,
#                                max_rri_sec,
#                                removal_warning_threshold,
#                                verbose,
#                                sub_id):
#     """
#     Align RRI values to ECG sample timeline using linear interpolation.
#     Removes RR intervals that are too long (e.g., due to missed beats).
    
#     Returns:
#         rri_interp_samples: array of interpolated RRI values (in samples), NaNs where not defined
#     """
#     # Compute RR intervals (in samples) and their center times
#     rri_values = np.diff(rpeaks)
#     rri_times = (rpeaks[:-1] + rpeaks[1:]) / 2

#     # Remove too long RR intervals (normaly 0 because handle in peak detection)
#     min_rri_samples = min_rri_sec * sfreq_ecg
#     max_rri_samples = max_rri_sec * sfreq_ecg
#     valid_mask = (rri_values >= min_rri_samples) & (rri_values <= max_rri_samples)
#     removed_ratio = 1.0 - np.mean(valid_mask) if len(valid_mask) > 0 else 0.0
#     if verbose:
#         print(f"Removed {removed_ratio * 100:.2f}% of RR intervals (> {max_rri_sec}s)")
#     #if removed_ratio > removal_warning_threshold:
#         #print(f"[WARNING] Sub {sub_id}: Removed {removed_ratio*100:.1f}% of RRIs ({min_rri_sec:.2f}-{max_rri_sec:.2f}s)")

#     rri_values_valid = rri_values[valid_mask]
#     if verbose:
#         print(f"Number of NaN in rri_values:{np.isnan(rri_values_valid).sum()}")
#     rri_times_valid = rri_times[valid_mask]

#     # Interpolate RRI values onto ECG sample timeline
#     ecg_samples = np.arange(ecg_len)
#     print("Lengths check:", len(rpeaks), len(rri_times), len(rri_values), len(rri_times_valid), len(rri_values_valid), len(ecg_samples))
#     if len(rri_values_valid) >= 2:
#         interp_func = interp1d(rri_times_valid,
#                                rri_values_valid,
#                                kind="linear",
#                                bounds_error=False,
#                                fill_value=np.nan)
#         rri_interp = interp_func(ecg_samples)
#     else:
#         print(f"[WARNING] Sub {sub_id}: Not enough valid RRI values for interpolation — all NaNs returned.")
#         rri_interp = np.full(ecg_len, np.nan)

#     # Debug info
#     n_nans = np.isnan(rri_interp).sum()
#     if verbose:
#         print(f"[WARNING] Sub {sub_id}: NaNs in interpolated RRI {n_nans} / {len(rri_interp)} "
#           f"({n_nans / len(rri_interp) * 100:.2f}%)")

#     return rri_interp

# def trim_rri_edr(rri_signal, edr_signal, windows_dict_ecg, sleep_onset_time, sfreq_ecg, config):

#     # Print NaN info for rri
#     if config.run.verbose:
#         rri_leading_nans = np.argmax(~np.isnan(rri_signal))
#         rri_trailing_nans = np.argmax(~np.isnan(rri_signal[::-1]))
#         rri_total_nans = np.isnan(rri_signal).sum()
#         rri_internal_nans = rri_total_nans - rri_leading_nans - rri_trailing_nans
#         if config.run.verbose:
#             print(f"RRI NaNs - leading: {rri_leading_nans}, internal: {rri_internal_nans}, trailing: {rri_trailing_nans}")

#     # Print NaN info for edr
#     if config.run.verbose:
#         edr_leading_nans = np.argmax(~np.isnan(edr_signal))
#         edr_trailing_nans = np.argmax(~np.isnan(edr_signal[::-1]))
#         edr_total_nans = np.isnan(edr_signal).sum()
#         edr_internal_nans = edr_total_nans - edr_leading_nans - edr_trailing_nans
#         print(f"EDR NaNs - leading: {edr_leading_nans}, internal: {edr_internal_nans}, trailing: {edr_trailing_nans}")

#     # Find overlapping valid region
#     valid = (~np.isnan(rri_signal)) & (~np.isnan(edr_signal))
#     if not valid.any():
#         raise ValueError("No overlapping valid data between rri and edr")
#     start_idx = np.argmax(valid)  # first True
#     end_idx = len(valid) - np.argmax(valid[::-1])  # last True + 1
#     if config.run.verbose:
#         print(f"Trim indices - start: {start_idx}, end: {end_idx} (length {end_idx - start_idx})")
    
#     # Trim signals
#     rri_trim = rri_signal[start_idx:end_idx]
#     edr_trim = edr_signal[start_idx:end_idx]
#     trim_offset = timedelta(seconds=start_idx / sfreq_ecg)
#     if config.run.verbose:
#         print(f"Number of NaN in rri after trim:{np.isnan(rri_trim).sum()}")
#         print(f"Number of NaN in edr after trim:{np.isnan(edr_trim).sum()}")
#         print(f"Trim Offset: {trim_offset}")

#         for stage, window_data in windows_dict_ecg.items():
#             if 'timestamps' in window_data and window_data['timestamps']:
#                 print(f"BEFORE trimming for stage: {stage}, First: {window_data['timestamps'][0]}, Last: {window_data['timestamps'][-1]}")

#     # Adjust windows dict
#     window_size_sec = config.analysis.window_size_min * 60
#     window_size_samples = int(window_size_sec * sfreq_ecg)
#     windows_dict_trim = copy.deepcopy(windows_dict_ecg)
#     for stage, window_data in windows_dict_trim.items():
#         if "timestamps" in window_data:
#             orig_timestamps = window_data["timestamps"]

#             # Convert timestamps back into sample indices (relative to original signal)
#             orig_indices = [
#                 int(round((ts - sleep_onset_time).total_seconds() * sfreq_ecg))
#                 for ts in orig_timestamps
#             ]

#             # Keep only indices within trimmed range
#             valid_indices = [
#                 i for i in orig_indices
#                 if start_idx <= i and i + window_size_samples <= end_idx
#             ]

#             # Convert valid indices back to timestamps (absolute time)
#             new_timestamps = [
#                 sleep_onset_time + timedelta(seconds=i / sfreq_ecg)
#                 for i in valid_indices
#             ]

#             # Update dictionary
#             window_data["timestamps"] = new_timestamps
#             window_data["count"] = len(new_timestamps)

#     if config.run.verbose:
#         for stage, window_data in windows_dict_trim.items():
#             if 'timestamps' in window_data and window_data['timestamps']:
#                 print(f"AFTER trimming for stage: {stage}, First: {window_data['timestamps'][0]}, Last: {window_data['timestamps'][-1]}")

#     return trim_offset, rri_trim, edr_trim, windows_dict_trim
   

# def downsample_rri_edr(rri_samples_trim,
#                        edr_samples_trim,
#                        sfreq_ecg,
#                        target_sfreq,
#                        verbose):
#     """
#     Downsample RRI and EDR signals from ECG sampling rate to a target frequency (e.g., 2 Hz).
#     Handles NaNs with linear interpolation and uses MNE's anti-aliasing-aware resample.
    
#     Returns:
#         rri_ds: downsampled RRI (in seconds)
#         edr_ds: downsampled EDR (in seconds)
#         time_ds: time vector (in seconds) for the downsampled signals
#     """
#     # RRi

#     # Fill NaNs via linear interpolation (constant edge padding)
#     rri_filled = pd.Series(rri_samples_trim).interpolate(limit_direction="both").to_numpy()
#     edr_filled = pd.Series(edr_samples_trim).interpolate(limit_direction="both").to_numpy()

#     # Optional: show number of NaNs before and after
#     if verbose:
#         print("RRI NaNs before interpolate:", np.isnan(rri_samples_trim).sum())
#         print("RRI NaNs after interpolate:", np.isnan(rri_filled).sum())
#         print("EDR NaNs before interpolate:", np.isnan(edr_samples_trim).sum())
#         print("EDR NaNs after interpolate: ", np.isnan(edr_filled).sum())

#     # Resample signals
#     rri_ds = mne.filter.resample(rri_filled, up=target_sfreq, down=sfreq_ecg)
#     edr_ds = mne.filter.resample(edr_filled, up=target_sfreq, down=sfreq_ecg)

#     # Time vector for resampled signals
#     time_ds = np.arange(len(rri_ds)) / target_sfreq

#     return rri_ds, edr_ds, time_ds

# def get_cpc_features_window(rri_seg, edr_seg, fs,
#                      freq_vl=[0, 0.01], freq_l=[0.01, 0.1], freq_h=[0.1, 0.4],
#                      use_max_two=True): 

#     rri_seg_sec = rri_seg / fs
#     # Compute cross-spectral coherence
#     ts = TimeSeries(np.array([rri_seg_sec, edr_seg]), sampling_rate=fs)
#     analyzer = MTCoherenceAnalyzer(ts)

#     freqs = analyzer.frequencies  # frequency bins
#     coh = analyzer.coherence[0, 1]  # coherence between RRI and EDR

#     # Cross spectral density (CSD)
#     # CPC is coherence × |CSD|^2, using multitaper method
#     csd = np.abs(mtm_cross_spectrum(analyzer.spectra[0], analyzer.spectra[1],
#                                     (analyzer.weights[0], analyzer.weights[1]),
#                                     sides='onesided')) ** 2

#     cpc = coh * csd  # final CPC values across frequency bins

#     vlfc = band_power(cpc, freqs, freq_vl, use_max_two)
#     lfc = band_power(cpc, freqs, freq_l, use_max_two)
#     hfc = band_power(cpc, freqs, freq_h, use_max_two)

#     # Avoid division by zero
#     ratio_l2h = lfc / hfc if hfc != 0 else np.nan
#     ratio_vl2lh = vlfc / (lfc + hfc) if (lfc + hfc) != 0 else np.nan
#     # Log and ratio dict (your existing CPC features)
#     log_vlfc = safe_log(vlfc)
#     log_lfc = safe_log(lfc)
#     log_hfc = safe_log(hfc)
#     log_h2l = log_hfc - log_lfc if not np.isnan(log_hfc) and not np.isnan(log_lfc) else np.nan
#     log_vl2lh = log_vlfc - safe_log(lfc + hfc)

#     cpc_feats = {
#         "pwr_VLFC": vlfc,
#         "pwr_LFC": lfc,
#         "pwr_HFC": hfc,
#         "ratio_L2H": ratio_l2h,
#         "ratio_VL2LH": ratio_vl2lh ,
#         "log_pwr_VLFC": log_vlfc,
#         "log_pwr_LFC": log_lfc,
#         "log_pwr_HFC": log_hfc,
#         "log_pwr_H2L": log_h2l,
#         "log_pwr_VL2LH": log_vl2lh,
#     }

#     return cpc_feats


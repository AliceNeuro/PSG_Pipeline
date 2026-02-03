import os 
from socketserver import ThreadingUnixStreamServer
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
import scipy.io as sio
from scipy.signal import resample, find_peaks, butter, filtfilt, savgol_filter
from scipy.stats import mode
from scipy.fft import fft, fftfreq

def extract_vb(row, tmp_dir_sub, resp_data, sleep_stages, verbose):
    psg_id = f"sub-{row['sub_id']}_ses-{row['session']}"
    sfreq_global = float(row['sfreq_global'])
    if not resp_data:
        if verbose:
            print(f"{psg_id}: No respiratory data found → skipping")
        return None
    
    good_signal = False
    
    # if "NASAL_PRESSURE" in resp_data.keys():
    #     full_resp = resp_data["NASAL_PRESSURE"]['full_signal']
    #     sfreq_resp = float(resp_data["NASAL_PRESSURE"]['sfreq_signal'])
    #     full_resp, bool_change_unit = check_resp_units(full_resp, psg_id)
    #     if bool_change_unit:
    #         print(f"[INFO] {psg_id} - 'NASAL_PRES': Unit change factor 10^6")
    #     full_resp = rescale_signal(sfreq_global, sfreq_resp, full_resp) 
    #     # change to say min 25 max sfreq global if in between no change !!
    #     full_resp = ensure_inspiration_up(full_resp)   
    #     if sanity_check(full_resp, sfreq_global, psg_id):
    #         good_signal = True

    if ("ABDOMINAL" in resp_data.keys()) and ("THORACIC" in resp_data.keys()) and not good_signal:
        abdo = resp_data["ABDOMINAL"]['signal'] # already cut to sleep only
        thor = resp_data["THORACIC"]['signal'] # already cut to sleep only
        fs_abd  = float(resp_data["ABDOMINAL"]["sfreq_signal"])
        fs_thor = float(resp_data["THORACIC"]["sfreq_signal"])
        # Require same fs; if not, stop
        if not np.isclose(fs_thor, fs_abd):
            if verbose:
                print(f"{psg_id}: THORACIC fs ({fs_thor}) != ABDOMINAL fs ({fs_abd}) -> cannot derive RIPFlow")
            return None
        
        # Check unit (MNE)
        abdo, bool_change_unit = check_resp_units(abdo, psg_id)
        if bool_change_unit:
            print(f"[INFO] {psg_id} - 'ABDOMINAL': Unit change factor 10^6")
        thor, bool_change_unit = check_resp_units(thor, psg_id)
        if bool_change_unit:
            print(f"[INFO] {psg_id} - 'THORACIC': Unit change factor 10^6")
        
        sfreq_resp = float(resp_data["THORACIC"]['sfreq_signal'])
        # Deducing RIPflow
        resp_signal, sfreq_resp = derive_ripflow_squared(thor, abdo, sfreq_resp)

        ### PRINT for debug
        if np.isnan(resp_signal).any():
            n_nan = np.isnan(resp_signal).sum()
            print(f"[CHECK] {psg_id}: ⚠️ NaNs detected in full_resp ({n_nan}/{resp_signal.size})")
        else:
            print(f"[CHECK] {psg_id}: ✅ no NaNs in full_resp")
        print("min:", np.nanmin(resp_signal))
        print("max:", np.nanmax(resp_signal))
        print("mean:", np.nanmean(resp_signal))
        print("median:", np.nanmedian(resp_signal))
        ### 

        if sanity_check(resp_signal, sfreq_global, psg_id):
            good_signal = True
        
    
    # if "THERM " in resp_data.keys() and not good_signal: 
    #     print(f"[IMPORTANT] {psg_id}: VB compute on THERM channel !")
    #     full_resp = resp_data["THERM"]['full_signal']
    #     sfreq_resp = float(resp_data["THERM"]['sfreq_signal'])
    #     full_resp = rescale_signal(sfreq_global, sfreq_resp, full_resp)
    #     full_resp = ensure_inspiration_up(full_resp)   
    #     if sanity_check(full_resp, sfreq_global, psg_id):
    #         good_signal = True
 
    # Check if full_resp OK
    if not good_signal:
        if verbose: 
            print(f"{psg_id}: Any Resp signal pass signal_check !")
        return None
    
    # Geat df_breath  
    breath_mat = tmp_dir_sub / f"breath_{psg_id}.mat"
    df_breath = get_breath_array(breath_mat, resp_signal, sfreq_resp, verbose)
    if df_breath.empty:
        if verbose:
            print(f"[WARNING] {psg_id}: No breaths detected → skipping VB extraction")
        return None

    # Sleep stages need ot match sfreq_resp
    n_samples = int(round(len(sleep_stages) * sfreq_resp / sfreq_global))
    sleep_stages = resample(sleep_stages , n_samples)
    
    sleep_stage_per_breath = []
    for idx, row in df_breath.iterrows():
        start_idx = int(row['insp_onset'])
        stop_idx = int(row['exp_offset']) + 1
        stages_slice = sleep_stages[start_idx:stop_idx]
        valid_stages = stages_slice[~np.isnan(stages_slice)]
        
        if len(valid_stages) == 0:
            if verbose:
                print(f"[WARNING] {psg_id}: No valid stages for breath idx:", idx, "Start:", start_idx, "Stop:", stop_idx)
            sleep_stage_per_breath.append(np.nan)
        else: # find the most frequent stage
            most_common_stage = mode(valid_stages, keepdims=True).mode[0]
            sleep_stage_per_breath.append(most_common_stage)
            
    df_breath['sleep_stage'] = sleep_stage_per_breath

    # Results per sleep stages
    results = {}
    sleep_periods = ['WN', 'SLEEP', 'NREM', 'N2N3', 'REM']

    for period in sleep_periods:
        if period == 'WN':
            df_period = df_breath  # all breaths
        elif period == 'SLEEP':
            df_period = df_breath[df_breath['sleep_stage'].isin([1,2,3,4])]
        elif period == 'NREM':
            df_period = df_breath[df_breath['sleep_stage'].isin([1,2,3])]
        elif period == 'N2N3':
            df_period = df_breath[df_breath['sleep_stage'].isin([1,2])]
        elif period == 'REM':
            df_period = df_breath[df_breath['sleep_stage'].isin([4])]

        # Only compute if enough breaths
        if len(df_period) >= 5:
            amp = df_period['normalized_amplitude']
            amp = np.clip(amp, 0, 200)
            bins = np.arange(0, 205, 5)
            hist, _ = np.histogram(amp, bins=bins)
            hist_percentage = (hist/len(amp)) * 100
            results[f'vb@{period}'] = np.sum(hist_percentage[:10]) # <= 50% 
        else:
            results[f'vb@{period}'] = np.nan
        
    print(results)

    return results


def rescale_signal(sfreq_global, sfreq_resp, full_signal):
    full_signal = np.asarray(full_signal, dtype=float)
    fs_old = float(sfreq_resp)
    fs_new = float(sfreq_global)

    if fs_old <= 0 or fs_new <= 0:
        raise ValueError("Sampling frequencies must be positive.")

    # --- No rescaling needed ---
    if np.isclose(fs_old, fs_new):
        return full_signal, fs_old

    ratio = fs_old / fs_new

    # --- Integer downsampling ---
    if fs_new < fs_old and np.isclose(ratio, round(ratio)):
        factor = int(round(ratio))
        full_resp = full_signal[::factor]

    # --- Upsampling OR non-integer downsampling ---
    else:
        n_samples = int(round(len(full_signal) * fs_new / fs_old))
        full_resp = resample(full_signal, n_samples)

    return full_resp


def derive_ripflow_squared(thor, abdo, sfreq_resp):
    # Input validation
    if len(thor) != len(abdo):
        raise ValueError("Thoracic and abdominal signals must have same length")
    
    # Compute volume signal and rescale to global sfreq
    volume = np.asarray(abdo, float) + np.asarray(thor, float)

    # Upsampling if fs < 25Hz
    if sfreq_resp < 25.0:
        n_samples = int(round(len(volume) * 25.0 / sfreq_resp))
        volume = resample(volume , n_samples)
        sfreq_resp = 25.0

    # Enhanced NaN handling with validation
    def _interp_nans(x):
        x = x.copy()
        nan_count = np.sum(~np.isfinite(x))
        if nan_count > 0:
            print(f"Warning: Interpolating {nan_count} NaN values ({nan_count/len(x)*100:.1f}%)")
        
        idx = np.arange(len(x))
        m = np.isfinite(x)
        if np.sum(m) == 0:
            print("Error: All values are NaN!")
            return np.zeros_like(x)
        if np.sum(m) < 10:
            print(f"Warning: Only {np.sum(m)} valid points for interpolation")
            
        x[~m] = np.interp(idx[~m], idx[m], x[m])
        return x
    
    volume = _interp_nans(volume)

    def lowpass_filter(signal, cutoff, fs, order=3):
        nyq = 0.5 * fs
        if cutoff >= nyq:
            print(f"Warning: Cutoff {cutoff} Hz >= Nyquist {nyq} Hz. Using {nyq*0.8} Hz")
            cutoff = nyq * 0.8
            
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return filtfilt(b, a, signal)
    
    volume_filtered = lowpass_filter(volume, cutoff=0.1, fs=sfreq_resp)

    # Derive flow using Savitzky-Golay
    dt = 1.0 / sfreq_resp
    ripflow = savgol_filter(volume_filtered, window_length=11, 
                           polyorder=3, deriv=1, delta=dt)

    # Clip
    # Signed square transform and standardization
    ripflow_sq = np.sign(ripflow) * (ripflow**2)

    return ripflow_sq, sfreq_resp


def check_resp_units(full_resp, psg_id, threshold_low=0.01, threshold_high=10.0):
    """
    Check if respiratory signal is in expected units.
    If the amplitude is too small (likely in volts), suggest scaling by 1e6.

    Parameters
    ----------
    full_resp : np.ndarray
        Respiratory signal.
    psg_id : str
        Identifier for logging.
    threshold_low : float
        Minimum expected peak-to-peak amplitude in "correct" units.
    threshold_high : float
        Maximum expected peak-to-peak amplitude in "correct" units.
    
    Returns
    -------
    scaled_resp : np.ndarray
        Signal scaled if necessary.
    scaled : bool
        True if signal was scaled.
    """
    ptp = np.ptp(full_resp)
    if ptp < threshold_low:
        print(f"[INFO] {psg_id}: signal amplitude very low (ptp={ptp:.5f}), scaling by 1e6")
        return full_resp * 1e6, True
    elif ptp > threshold_high:
        print(f"[INFO] {psg_id}: signal amplitude very high (ptp={ptp:.2f}), check units")
        return full_resp, False
    else:
        # likely correct units
        return full_resp, False
    

def sanity_check(full_resp, sfreq_resp, psg_id, ptp_threshold=0.01, noise_threshold=0.5):
    """
    Minimal sanity check for a respiratory signal.

    Checks:
        1. Signal is not too noisy (FFT-based)
        2. Peak-to-peak amplitude is reasonable (empirical threshold or z-score)
    """
    full_resp = full_resp.flatten()
    
    # Basic validity checks
    if full_resp is None or len(full_resp) < 100:
        print(f"[WARNING] {psg_id}: resp signal too short -> skipping")
        return False
    
    if np.isnan(full_resp).all():
        print(f"[WARNING] {psg_id}: resp signal all NaN -> skipping")
        return False
    
    if np.any(np.isnan(full_resp)):
        print(f"[WARNING] {psg_id}: resp signal contains NaN -> linear interpolation")
        nans = np.isnan(full_resp)
        if nans.all():
            return False
        full_resp[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), full_resp[~nans])

    if sfreq_resp <= 0 or np.isnan(sfreq_resp):
        print(f"[WARNING] {psg_id}: invalid fs -> skipping")
        return False

    # 1. FFT-based noise check
    N = len(full_resp)
    yf = fft(full_resp)
    xf = fftfreq(N, 1 / sfreq_resp)

    # Compute signal-to-noise ratio: ratio of power in 0.1-0.5 Hz (breathing) vs >0.5 Hz (noise)
    breathing_band = (xf > 0.1) & (xf < 0.5)
    noise_band = xf > 0.5
    signal_power = np.sum(np.abs(yf[breathing_band])**2)
    noise_power = np.sum(np.abs(yf[noise_band])**2)
    
    if noise_power == 0:
        snr = np.inf
    else:
        snr = signal_power / noise_power

    if snr < noise_threshold:
        print(f"[WARNING] {psg_id}: resp signal too noisy (SNR={snr:.2f}) -> skipping")
        return False

    # 2. Peak-to-peak amplitude check
    ptp = np.ptp(full_resp)
    if ptp < ptp_threshold:
        print(f"[WARNING] {psg_id}: signal too flat (ptp={ptp:.5f}) -> skipping")
        return False

    # Passed all sanity checks
    return True


def ensure_inspiration_up(flow):
    """
    Orient 'flow' by making the positive side have larger median magnitude than the negative side.
    """
    x = np.asarray(flow, float)
    x = x - np.nanmedian(x)  # center to reduce bias from drift

    pos = x[x > 0]
    neg = -x[x < 0]  # magnitudes of negative values

    # Require some data on both sides
    if len(pos) == 0 or len(neg) == 0:
        return x

    if np.nanmedian(pos) < np.nanmedian(neg):
        x = -x
    return x


def get_breath_array(breath_mat, full_resp, sfreq_resp, verbose):
    sio.savemat(breath_mat, {'nas_pres': full_resp, 
                            'fs': sfreq_resp, 
                            'opts': {'plotFig': 0}})

    project_src_root = Path(__file__).resolve().parents[1] 
    mat_breath_script = project_src_root / "external_tools/matlab/breath_table/call_breathtable.m"

    # Build MATLAB command
    command = [
        "/usr/local/matlab/R2024a/bin/matlab", # "/usr/local/matlab/R2024a/bin/matlab", # "/Applications/MATLAB_R2025a.app/bin/matlab"
        "-nojvm",
        "-nosplash",
        "-nodesktop",
        "-softwareopengl",
        "-batch",
        f'addpath("{mat_breath_script.parent}"); call_breathtable("{breath_mat}")'
    ]
    try:
        subprocess.run(
                    command,
                    check=True,
                    stdout=None if verbose else subprocess.DEVNULL,
                    stderr=None if verbose else subprocess.DEVNULL
                )
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] MATLAB breath detection failed for file: {breath_mat}: {e}")
        return pd.DataFrame()  # return empty DataFrame on failure


    # Load output from MATLAB
    breath_data = sio.loadmat(breath_mat)
    breath_array = breath_data['breath_array']

    colnames = ['breath_id', 'insp_onset', 'peak', 'insp_breath_offset',
            'exp_peak', 'exp_onset', 'exp_offset', 'fs',
            'RespiratoryRate', 'val_fl', 'Ttot', 'normalized_amplitude']

    df_breath = pd.DataFrame(breath_array, columns=colnames)

    return df_breath
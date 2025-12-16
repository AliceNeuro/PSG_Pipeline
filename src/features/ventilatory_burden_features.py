import os 
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
import scipy.io as sio
from scipy.signal import resample, find_peaks, butter, filtfilt, savgol_filter
from scipy.stats import mode
from scipy.fft import fft, fftfreq

def extract_vb(row, tmp_dir_sub, resp_data, full_sleep_stages, verbose):
    psg_id = f"sub-{row['sub_id']}_ses-{row['session']}"
    sfreq_global = float(row['sfreq_global'])
    if not resp_data:
        if verbose:
            print(f"{psg_id}: No respiratory data found → skipping")
        return None
    
    good_signal = False
    
    if "NASAL_PRESSURE" in resp_data.keys():
        full_resp = resp_data["NASAL_PRESSURE"]['full_signal']
        sfreq_resp = float(resp_data["NASAL_PRESSURE"]['sfreq_signal'])
        full_resp, bool_change_unit = check_resp_units(full_resp, psg_id)
        if bool_change_unit:
            print(f"[INFO] {psg_id} - 'NASAL_PRES': Unit change factor 10^6")
        full_resp, sfreq_resp, full_sleep_stages = rescale_signal(sfreq_global, sfreq_resp, full_resp, full_sleep_stages, do_stages=True)
        if sanity_check(full_resp, sfreq_resp, psg_id):
            good_signal = True

    if ("ABDOMINAL" in resp_data.keys()) and ("THORACIC" in resp_data.keys()) and not good_signal:
        full_abdo = resp_data["ABDOMINAL"]['full_signal']
        full_thor = resp_data["THORACIC"]['full_signal']
        fs_abd  = float(resp_data["ABDOMINAL"]["sfreq_signal"])
        fs_thor = float(resp_data["THORACIC"]["sfreq_signal"])
        # Require same fs; if not, stop
        if not np.isclose(fs_thor, fs_abd):
            if verbose:
                print(f"{psg_id}: THORACIC fs ({fs_thor}) != ABDOMINAL fs ({fs_abd}) -> cannot derive RIPFlow")
            return None
        
        # Check unit (MNE)
        full_abdo, bool_change_unit = check_resp_units(full_abdo, psg_id)
        if bool_change_unit:
            print(f"[INFO] {psg_id} - 'ABDOMINAL': Unit change factor 10^6")
        full_thor, bool_change_unit = check_resp_units(full_thor, psg_id)
        if bool_change_unit:
            print(f"[INFO] {psg_id} - 'THORACIC': Unit change factor 10^6")
        
        sfreq_resp = float(resp_data["THORACIC"]['sfreq_signal'])
        full_abdo, sfreq_resp, _ = rescale_signal(sfreq_global, sfreq_resp, full_abdo, full_sleep_stages, do_stages=False)
        full_thor, sfreq_resp, full_sleep_stages = rescale_signal(sfreq_global, sfreq_resp, full_thor, full_sleep_stages, do_stages=True)
         
        # Deducing RIPflow
        ripflow_sq = derive_ripflow_squared(full_thor, full_abdo, sfreq_resp)
        full_resp = ripflow_sq 
        if sanity_check(full_resp, sfreq_resp, psg_id):
            good_signal = True
        
    
    if "THERM " in resp_data.keys() and not good_signal: # RESTRAT HERE !!
        print(f"[IMPORTANT] {psg_id}: VB compute on THERM channel !")
        full_resp = resp_data["THERM"]['full_signal']
        sfreq_resp = float(resp_data["THERM"]['sfreq_signal'])
        full_resp, sfreq_resp, full_sleep_stages = rescale_signal(sfreq_global, sfreq_resp, full_resp, full_sleep_stages, do_stages=True)

    # Check if full_resp OK
    if not good_signal:
        if verbose: 
            print(f"{psg_id}: Any Resp signal pass signal_check !")
        return None
    
    # Need to have the inspiraion up
    full_resp = ensure_inspiration_up(full_resp)
    
    # Geat df_breath  
    breath_mat = tmp_dir_sub / f"breath_{psg_id}.mat"
    df_breath = get_breath_array(breath_mat, full_resp, sfreq_resp, verbose)
    if df_breath.empty:
        if verbose:
            print(f"[WARNING] {psg_id}: No breaths detected → skipping VB extraction")
        return None
    # Reduce df_breath 
    sleep_ids = np.where(np.isin(full_sleep_stages, [1, 2, 3, 4]))[0]
    start_index = sleep_ids[0]
    end_index = sleep_ids[-1] + 1 # include the last index
    df_breath = df_breath[(df_breath['insp_onset'] >= start_index) & (df_breath['exp_offset'] <= end_index)]

    # Add the most frequent sleep stage per breath
    sleep_stage_per_breath = []
    for idx, row in df_breath.iterrows():
        start_idx = int(row['insp_onset'])
        stop_idx = int(row['exp_offset']) + 1
        stages_slice = full_sleep_stages[start_idx:stop_idx]
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

    return results


def rescale_signal(sfreq_global, sfreq_resp, full_signal, full_sleep_stages, do_stages):
    """
    Rescale a signal to match a target sampling frequency.
    Always returns a tuple: (rescaled_signal, rescaled_stages_or_None)
    """
    if sfreq_resp < 25.0: 
            sfreq_resp = 25.0

    ratio = sfreq_global / sfreq_resp
    
    full_stages_rescaled = None
    
    if np.isclose(ratio, round(ratio)):
        factor = int(round(ratio))
        full_resp = full_signal[::factor]
        if do_stages and full_sleep_stages is not None:
            full_stages_rescaled = full_sleep_stages[::factor]
    else:
        n_samples = int(round(len(full_signal) * sfreq_resp / sfreq_global))
        full_resp = resample(full_signal, n_samples)
        if do_stages and full_sleep_stages is not None:
            idx = np.linspace(0, len(full_sleep_stages) - 1, n_samples).astype(int)
            full_stages_rescaled = full_sleep_stages[idx]
    
    return full_resp, sfreq_resp, full_stages_rescaled


def derive_ripflow_squared(
        thor, abd, fs,
        bandpass=(0.02, 2.0),   # Hz: remove drift and high-frequency noise
        deriv_win_sec=2.0,      # Savitzky–Golay window (seconds)
        deriv_poly=3,           # Savitzky–Golay polynomial order
        ensure_insp_positive=True):
    """
    Derive inspiratory-only squared airflow (RIPFlow^2) from thoracic and abdominal RIP belts.

    Inputs:
      thor, abd : 1D numpy arrays (same length) of RIP belt signals
      fs        : sampling rate (Hz)
      bandpass  : (low, high) bandpass cutoffs in Hz for volume surrogate
      deriv_win_sec : window length for Savitzky–Golay derivative (in seconds)
      deriv_poly    : polynomial order for Savitzky–Golay
      ensure_insp_positive : if True, flips sign so inspiration is positive

    Returns:
      ripflow_sq : 1D numpy array, inspiratory-only squared derived airflow
    """
    thor = np.asarray(thor, dtype=float)
    abd  = np.asarray(abd, dtype=float)
    if thor.shape != abd.shape:
        raise ValueError("Thoracic and abdominal arrays must have the same shape.")
    if len(thor) < int(fs):  # need at least ~1 second of data
        raise ValueError("Signal too short to derive RIPFlow.")

    # Handle NaNs by simple interpolation
    def _interp_nans(x):
        x = x.copy()
        idx = np.arange(len(x))
        m = np.isfinite(x)
        if np.sum(m) == 0:
            return np.zeros_like(x)
        x[~m] = np.interp(idx[~m], idx[m], x[m])
        return x
    thor = _interp_nans(thor)
    abd  = _interp_nans(abd)

    # Sum belts to get a volume surrogate
    vol = thor + abd
    vol = vol - np.nanmean(vol)  # de-mean

    # Bandpass filter the volume surrogate
    if bandpass is not None:
        nyq = 0.5 * fs
        low = max(bandpass[0] / nyq, 1e-6)
        high = min(bandpass[1] / nyq, 0.999999)
        if low >= high:
            raise ValueError("Invalid bandpass cutoffs.")
        b, a = butter(4, [low, high], btype='bandpass')
        # filtfilt needs enough points; if too short, skip filtering
        if len(vol) > max(len(a), len(b)) * 3:
            vol = filtfilt(b, a, vol)

    # Derive flow via Savitzky–Golay differentiation
    win = int(round(deriv_win_sec * fs))
    if win < (deriv_poly + 2):
        win = deriv_poly + 3
    if win % 2 == 0:
        win += 1
    flow = savgol_filter(vol, window_length=win, polyorder=deriv_poly,
                         deriv=1, delta=1/fs)

    # Ensure inspiration is positive
    if ensure_insp_positive:
        pos_area = np.sum(np.maximum(flow, 0.0))
        neg_area = -np.sum(np.minimum(flow, 0.0))
        if pos_area < neg_area:
            flow = -flow

    # Keep only inspiratory phase and square to mimic nasal-pressure-derived airflow
    flow_insp = np.maximum(flow, 0.0)
    ripflow_sq = flow_insp ** 2

    return ripflow_sq


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


def ensure_inspiration_up(full_resp):
    """
    Check polarity of respiratory signal.
    Flip if necessary so inspiration is upward.
    """
    full_resp = full_resp.flatten()
    
    # Heuristic: compare mean of first derivative during peaks
    deriv = np.diff(full_resp)
    if np.mean(deriv[:len(deriv)//2]) < 0:
        # Signal likely inverted
        full_resp = -full_resp
    return full_resp
    


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
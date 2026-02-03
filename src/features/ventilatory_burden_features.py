import os 
from socketserver import ThreadingUnixStreamServer
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
import scipy.io as sio
import scipy.signal as signal
import scipy.sparse as sp
from scipy.optimize import minimize
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
    
    if "NASAL_PRESSURE" in resp_data.keys():
        resp_signal = resp_data["NASAL_PRESSURE"]['signal']
        sfreq_resp = float(resp_data["NASAL_PRESSURE"]['sfreq_signal'])
        resp_signal, bool_change_unit = check_resp_units(resp_signal, psg_id)
        if bool_change_unit:
            print(f"[INFO] {psg_id} - 'NASAL_PRES': Unit change factor 10^6")
        
        # Upsampling if fs < 32Hz
        min_sfreq = 32.0  
        if sfreq_resp < min_sfreq:
            n_samples = int(round(len(resp_signal) * min_sfreq / sfreq_resp))
            resp_signal = resample(resp_signal , n_samples)
            sfreq_resp = min_sfreq

        # Apply DORIS de-clipper
        doris = DORISDeClipper(lambda_smooth=1.0, lambda_energy=0.01)
        decliped_resp_signal = doris.preprocess_ripflow(
            resp_signal,
            auto_detect=True,
            threshold_percentile=98.0  
        )

        # Breath-to-breath detrending
        breath_to_breath_detrender = BreathToBreathDetrend(sampling_rate=sfreq_resp)  
        detrend_resp_signal = breath_to_breath_detrender.preprocess_baseline_removal(decliped_resp_signal)

        # Sanity Check 
        if sanity_check(detrend_resp_signal, sfreq_resp, psg_id):
            good_signal = True


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
        resp_signal, sfreq_resp = derive_ripflow(thor, abdo, sfreq_resp)

        # # Apply DORIS de-clipper
        # doris = DORISDeClipper(lambda_smooth=1.0, lambda_energy=0.01)
        # decliped_resp_signal = doris.preprocess_ripflow(
        #     resp_signal,
        #     auto_detect=True,
        #     threshold_percentile=98.0  
        # )

        # # Breath-to-breath detrending
        # breath_to_breath_detrender = BreathToBreathDetrend(sampling_rate=sfreq_resp)  
        # detrend_resp_signal = breath_to_breath_detrender.preprocess_baseline_removal(decliped_resp_signal)
        
        # # Sign square results
        # detrend_resp_signal = np.sign(detrend_resp_signal) * (detrend_resp_signal**2)

        # Sanity Check 
        results_check = sanity_check(resp_signal, sfreq_resp, psg_id)
        if results_check['passed']:
            good_signal = True
 
    # Check that we have a good airflow signal to compute VB
    # if not good_signal:
    #     if verbose: 
    #         print(f"{psg_id}: Any Resp signal pass signal_check !")
    #     return None
    
    # Geat df_breath  
    breath_mat = tmp_dir_sub / f"breath_{psg_id}.mat"
    df_breath = get_breath_array(breath_mat, resp_signal, sfreq_resp, verbose)
    if df_breath.empty:
        if verbose:
            print(f"[WARNING] {psg_id}: No breaths detected → skipping VB extraction")
        return None

    # Compute VB
    results = {}
    if len(df_breath) >= 5:
        amp = df_breath['normalized_amplitude']
        amp = np.clip(amp, 0, 200)
        bins = np.arange(0, 205, 5)
        hist, _ = np.histogram(amp, bins=bins)
        hist_percentage = (hist/len(amp)) * 100
        results['VB'] = np.sum(hist_percentage[:10]) # <= 50% 
    else:
        results['VB'] = np.nan

    print(results)

    # # Sleep stages need to match sfreq_resp
    # print("Length before:", len(sleep_stages))
    # n_samples = int(round(len(sleep_stages) * sfreq_resp / sfreq_global))
    # sleep_stages = resample(sleep_stages , n_samples)
    # print("Length after:", len(sleep_stages))
    # print(sleep_stages.mean())

    # sleep_stage_per_breath = []
    # for idx, row in df_breath.iterrows():
    #     start_idx = int(row['insp_onset'])
    #     stop_idx = int(row['exp_offset']) + 1
    #     stages_slice = sleep_stages[start_idx:stop_idx]
    #     valid_stages = stages_slice[~np.isnan(stages_slice)]
        
    #     if len(valid_stages) == 0:
    #         #if verbose:
    #             #print(f"[WARNING] {psg_id}: No valid stages for breath idx:", idx, "Start:", start_idx, "Stop:", stop_idx)
    #         sleep_stage_per_breath.append(np.nan)
    #     else: # find the most frequent stage
    #         most_common_stage = mode(valid_stages, keepdims=True).mode[0]
    #         sleep_stage_per_breath.append(most_common_stage)
            
    # df_breath['sleep_stage'] = sleep_stage_per_breath

    # # Results per sleep stages
    # results = {}
    # sleep_periods = ['WN', 'SLEEP', 'NREM', 'N2N3', 'REM']

    # for period in sleep_periods:
    #     if period == 'WN':
    #         df_period = df_breath  # all breaths
    #     elif period == 'SLEEP':
    #         df_period = df_breath[df_breath['sleep_stage'].isin([1,2,3,4])]
    #     elif period == 'NREM':
    #         df_period = df_breath[df_breath['sleep_stage'].isin([1,2,3])]
    #     elif period == 'N2N3':
    #         df_period = df_breath[df_breath['sleep_stage'].isin([1,2])]
    #     elif period == 'REM':
    #         df_period = df_breath[df_breath['sleep_stage'].isin([4])]

    #     # Only compute if enough breaths
    #     if len(df_period) >= 5:
    #         amp = df_period['normalized_amplitude']
    #         amp = np.clip(amp, 0, 200)
    #         bins = np.arange(0, 205, 5)
    #         hist, _ = np.histogram(amp, bins=bins)
    #         hist_percentage = (hist/len(amp)) * 100
    #         results[f'vb@{period}'] = np.sum(hist_percentage[:10]) # <= 50% 
    #     else:
    #         results[f'vb@{period}'] = np.nan

    return results


def derive_ripflow(thor, abdo, sfreq_resp):
    # Input validation
    if len(thor) != len(abdo):
        raise ValueError("Thoracic and abdominal signals must have same length")
    
    # Compute volume signal and rescale to global sfreq
    volume = np.asarray(abdo, float) + np.asarray(thor, float)

    # Upsampling if fs < 32Hz
    min_sfreq = 32.0  
    if sfreq_resp < min_sfreq:
        n_samples = int(round(len(volume) * min_sfreq / sfreq_resp))
        volume = resample(volume , n_samples)
        sfreq_resp = min_sfreq

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
    
    # Check if need to flip the signal
    #is_inverted = correct_polarity(ripflow, sfreq_resp)
    #if is_inverted:
    #    print("Signal is inverted")
    ripflow = -ripflow

    # Clip signal before squared
    threshold_percentile = 98
    threshold = np.percentile(np.abs(ripflow), threshold_percentile)
    ripflow = np.clip(ripflow, -threshold, threshold)

    # Signe squared transform
    ripflow_sq = np.sign(ripflow) * (ripflow**2)

    # Normalize
    mean_val = np.nanmean(ripflow_sq)
    std_val = np.nanstd(ripflow_sq)
    std_val = max(std_val, 1e-8)
    ripflow_sq = (ripflow_sq - mean_val) / std_val

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

def correct_polarity(signal_data, sfreq_resp):
    """
    Ensure inspiration is upward (positive direction)
    """
    min_breath_duration = 1.0  # seconds
    max_breath_duration = 15.0  # seconds
    
    signal_data = np.array(signal_data)
    
    # Remove DC component for analysis
    detrended = signal_data - np.mean(signal_data)
    
    # Find peaks (potential inspirations) and troughs (potential expirations)
    peaks, _ = signal.find_peaks(detrended, height=np.std(detrended)*0.5, 
                                distance=int(sfreq_resp * min_breath_duration))
    troughs, _ = signal.find_peaks(-detrended, height=np.std(detrended)*0.5, 
                                    distance=int(sfreq_resp * min_breath_duration))
    
    # Calculate breath characteristics for both orientations
    if len(peaks) > 0 and len(troughs) > 0:
        # Current orientation: peaks as inspiration
        peak_prominences = signal.peak_prominences(detrended, peaks)[0]
        trough_prominences = signal.peak_prominences(-detrended, troughs)[0]
        
        # Measure breath regularity (coefficient of variation)
        if len(peaks) > 1:
            peak_intervals = np.diff(peaks) / sfreq_resp
            peak_cv = np.std(peak_intervals) / np.mean(peak_intervals)
        else:
            peak_cv = float('inf')
            
        if len(troughs) > 1:
            trough_intervals = np.diff(troughs) / sfreq_resp
            trough_cv = np.std(trough_intervals) / np.mean(trough_intervals)
        else:
            trough_cv = float('inf')
        
        # Decision criteria:
        # 1. Higher average prominence
        # 2. More regular intervals (lower CV)
        # 3. More physiological breathing rate (8-30 breaths/min)
        
        avg_peak_prominence = np.mean(peak_prominences)
        avg_trough_prominence = np.mean(trough_prominences)
        
        # Calculate breathing rates
        if len(peaks) > 1:
            peak_rate = 60 / np.mean(peak_intervals)
        else:
            peak_rate = 0
            
        if len(troughs) > 1:
            trough_rate = 60 / np.mean(trough_intervals)
        else:
            trough_rate = 0
        
        # Score each orientation
        peak_score = 0
        trough_score = 0
        
        # Prominence scoring
        if avg_peak_prominence > avg_trough_prominence:
            peak_score += 1
        else:
            trough_score += 1
        
        # Regularity scoring
        if peak_cv < trough_cv:
            peak_score += 1
        else:
            trough_score += 1
        
        # Physiological rate scoring (8-30 breaths/min)
        if 8 <= peak_rate <= 30:
            peak_score += 1
        if 8 <= trough_rate <= 30:
            trough_score += 1
        
        # Decision
        if trough_score > peak_score: # Troughs are better inspirations
            is_inverted = True
        else: # Peaks are inspirations, keep as is
            is_inverted = False
    else: # Fallback: assume signal is correct
        is_inverted = False
    
    return is_inverted


def sanity_check(full_resp, sfreq_resp, psg_id, ptp_threshold=0.01, noise_threshold=0.5):
    """
    Minimal sanity check for a respiratory signal.
    
    Checks:
    1. Signal is not too noisy (FFT-based)
    2. Peak-to-peak amplitude is reasonable (empirical threshold or z-score)
    
    Parameters:
    -----------
    full_resp : array-like
        The respiratory signal data
    sfreq_resp : float
        Sampling frequency of the respiratory signal in Hz
    psg_id : str or int
        PSG identifier for logging/debugging purposes
    ptp_threshold : float, default=0.01
        Minimum acceptable peak-to-peak amplitude threshold
    noise_threshold : float, default=0.5
        Maximum acceptable noise ratio (high frequency power / total power)
    
    Returns:
    --------
    dict : Dictionary containing check results
        - 'passed': bool, True if all checks passed
        - 'noise_check': bool, True if noise check passed
        - 'amplitude_check': bool, True if amplitude check passed
        - 'noise_ratio': float, ratio of high frequency to total power
        - 'ptp_amplitude': float, peak-to-peak amplitude
        - 'psg_id': identifier passed in
    """
    
    # Convert to numpy array and handle potential issues
    resp_signal = np.asarray(full_resp)
    
    # Initialize results dictionary
    results = {
        'passed': False,
        'noise_check': False,
        'amplitude_check': False,
        'noise_ratio': np.nan,
        'ptp_amplitude': np.nan,
        'psg_id': psg_id
    }
    
    # Basic signal validation
    if len(resp_signal) == 0:
        print(f"PSG {psg_id}: Empty respiratory signal")
        return results
    
    if np.all(np.isnan(resp_signal)) or np.all(resp_signal == 0):
        print(f"PSG {psg_id}: Signal contains only NaN or zero values")
        return results
    
    # Remove NaN values for analysis
    valid_signal = resp_signal[~np.isnan(resp_signal)]
    
    if len(valid_signal) < 10:  # Need minimum samples for analysis
        print(f"PSG {psg_id}: Too few valid samples for analysis")
        return results
    
    try:
        # Check 1: FFT-based noise assessment
        # Compute FFT
        n_samples = len(valid_signal)
        fft_signal = fft(valid_signal)
        freqs = fftfreq(n_samples, 1/sfreq_resp)
        
        # Get power spectrum (magnitude squared)
        power_spectrum = np.abs(fft_signal)**2
        
        # Define frequency bands
        # Respiratory frequency typically 0.1-0.5 Hz (6-30 breaths/min)
        # High frequency noise: above 2 Hz for respiratory signals
        respiratory_band = (freqs >= 0.05) & (freqs <= 1.0)
        high_freq_band = freqs > 2.0
        
        # Calculate power in different bands
        total_power = np.sum(power_spectrum[freqs >= 0])
        high_freq_power = np.sum(power_spectrum[high_freq_band])
        
        # Noise ratio: high frequency power / total power
        noise_ratio = high_freq_power / total_power if total_power > 0 else 1.0
        results['noise_ratio'] = noise_ratio
        
        # Noise check passes if noise ratio is below threshold
        results['noise_check'] = noise_ratio <= noise_threshold
        
        # Check 2: Peak-to-peak amplitude assessment
        ptp_amplitude = np.ptp(valid_signal)  # peak-to-peak amplitude
        results['ptp_amplitude'] = ptp_amplitude
        
        # Amplitude check passes if peak-to-peak is above threshold
        results['amplitude_check'] = ptp_amplitude >= ptp_threshold
        
        # Overall pass: both checks must pass
        results['passed'] = results['noise_check'] and results['amplitude_check']
        
        # Log results if checks fail
        if not results['passed']:
            fail_reasons = []
            if not results['noise_check']:
                fail_reasons.append(f"high noise ratio ({noise_ratio:.3f} > {noise_threshold})")
            if not results['amplitude_check']:
                fail_reasons.append(f"low amplitude ({ptp_amplitude:.3f} < {ptp_threshold})")
            
            print(f"PSG {psg_id}: Sanity check failed - {', '.join(fail_reasons)}")
    
    except Exception as e:
        print(f"PSG {psg_id}: Error during sanity check - {str(e)}")
        return results
    
    return results


def sanity_check_old(full_resp, sfreq_resp, psg_id, ptp_threshold=0.01, noise_threshold=0.5):
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
                    stdout=None,
                    stderr=None
                    # stdout=None if verbose else subprocess.DEVNULL,
                    # stderr=None if verbose else subprocess.DEVNULL
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



class DORISDeClipper:
    def __init__(self, lambda_smooth=1.0, lambda_energy=0.1):
        """
        DORIS: De-clipping using Optimization with Regularized Interpolation and Smoothness
        
        Parameters:
        -----------
        lambda_smooth : float
            Regularization parameter for smoothness constraint
        lambda_energy : float
            Regularization parameter for finite energy constraint
        """
        self.lambda_smooth = lambda_smooth
        self.lambda_energy = lambda_energy
    
    def detect_clipping(self, signal, threshold_percentile=99.5):
        """
        Detect clipped regions in the signal
        """
        threshold = np.percentile(np.abs(signal), threshold_percentile)
        clipped_mask = np.abs(signal) >= threshold
        return clipped_mask, threshold
    
    def create_smoothness_matrix(self, n):
        """
        Create second-order difference matrix for smoothness constraint
        """
        # Create second-order difference matrix
        diags = [1, -2, 1]
        offsets = [0, 1, 2]
        D2 = sp.diags(diags, offsets, shape=(n-2, n), format='csr')
        return D2
    
    def objective_function(self, x, y_obs, clipped_mask, D2):
        """
        Convex objective function for DORIS optimization
        
        Parameters:
        -----------
        x : array
            Signal to optimize
        y_obs : array
            Observed (clipped) signal
        clipped_mask : boolean array
            Mask indicating clipped samples
        D2 : sparse matrix
            Second-order difference matrix
            
        Returns:
        --------
        cost : float
            Objective function value
        """
        # Data fidelity term (only for non-clipped samples)
        data_term = np.sum((x[~clipped_mask] - y_obs[~clipped_mask])**2)
        
        # Smoothness term (second-order derivative)
        smooth_term = np.sum((D2.dot(x))**2)
        
        # Energy term
        energy_term = np.sum(x**2)
        
        # Combined objective
        cost = data_term + self.lambda_smooth * smooth_term + self.lambda_energy * energy_term
        
        return cost
    
    def gradient_function(self, x, y_obs, clipped_mask, D2):
        """
        Gradient of the objective function
        
        Parameters:
        -----------
        x : array
            Signal to optimize
        y_obs : array
            Observed (clipped) signal
        clipped_mask : boolean array
            Mask indicating clipped samples
        D2 : sparse matrix
            Second-order difference matrix
            
        Returns:
        --------
        grad : array
            Gradient vector
        """
        n = len(x)
        grad = np.zeros(n)
        
        # Gradient of data fidelity term
        grad[~clipped_mask] = 2 * (x[~clipped_mask] - y_obs[~clipped_mask])
        
        # Gradient of smoothness term
        grad += 2 * self.lambda_smooth * D2.T.dot(D2.dot(x))
        
        # Gradient of energy term
        grad += 2 * self.lambda_energy * x
        
        return grad
    
    def linear_interpolate_clipped(self, signal, clipped_mask):
        """
        Linear interpolation initialization for clipped regions
        
        Parameters:
        -----------
        signal : array
            Clipped signal
        clipped_mask : boolean array
            Clipping mask
            
        Returns:
        --------
        interpolated : array
            Signal with linear interpolation in clipped regions
        """
        interpolated = signal.copy()
        
        # Find clipped segments
        clipped_indices = np.where(clipped_mask)[0]
        
        if len(clipped_indices) == 0:
            return interpolated
        
        # Group consecutive clipped indices
        segments = []
        start = clipped_indices[0]
        
        for i in range(1, len(clipped_indices)):
            if clipped_indices[i] - clipped_indices[i-1] > 1:
                segments.append((start, clipped_indices[i-1]))
                start = clipped_indices[i]
        segments.append((start, clipped_indices[-1]))
        
        # Interpolate each segment
        for start_idx, end_idx in segments:
            # Find valid boundary points
            left_idx = max(0, start_idx - 1)
            right_idx = min(len(signal) - 1, end_idx + 1)
            
            # Ensure we have valid boundary values
            while left_idx >= 0 and clipped_mask[left_idx]:
                left_idx -= 1
            while right_idx < len(signal) and clipped_mask[right_idx]:
                right_idx += 1
            
            # Handle edge cases
            if left_idx < 0 and right_idx >= len(signal):
                interpolated[start_idx:end_idx+1] = 0  # Default to zero
            elif left_idx < 0:
                interpolated[start_idx:end_idx+1] = signal[right_idx]
            elif right_idx >= len(signal):
                interpolated[start_idx:end_idx+1] = signal[left_idx]
            else:
                # Linear interpolation
                x_vals = np.arange(start_idx, end_idx + 1)
                y_vals = np.interp(x_vals, [left_idx, right_idx], 
                                 [signal[left_idx], signal[right_idx]])
                interpolated[start_idx:end_idx+1] = y_vals
        
        return interpolated
    
    def declip_signal(self, signal, clipped_mask=None, threshold=None):
        """
        De-clip the airflow signal using DORIS algorithm
        
        Parameters:
        -----------
        signal : array-like
            Clipped airflow signal
        clipped_mask : boolean array, optional
            Pre-computed clipping mask
        threshold : float, optional
            Clipping threshold
            
        Returns:
        --------
        reconstructed_signal : array
            De-clipped signal
        clipped_mask : boolean array
            Mask of clipped regions
        """
        signal = np.array(signal)
        n = len(signal)
        
        # Detect clipping if not provided
        if clipped_mask is None:
            clipped_mask, threshold = self.detect_clipping(signal, threshold)
        
        # If no clipping detected, return original signal
        if not np.any(clipped_mask):
            return signal, clipped_mask
        
        # Create smoothness matrix
        D2 = self.create_smoothness_matrix(n)
        
        # Initial guess: linear interpolation of clipped regions
        x0 = self.linear_interpolate_clipped(signal, clipped_mask)
        
        # Define objective and gradient functions
        def obj_func(x):
            return self.objective_function(x, signal, clipped_mask, D2)
        
        def grad_func(x):
            return self.gradient_function(x, signal, clipped_mask, D2)
        
        # Optimize using L-BFGS-B (good for smooth problems)
        result = minimize(
            obj_func,
            x0,
            method='L-BFGS-B',
            jac=grad_func,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )
        
        return result.x, clipped_mask

    def preprocess_ripflow(self, ripflow_signal, auto_detect=True, threshold_percentile=99.5):
        """
        Main preprocessing function for RIPflow signals
        
        Parameters:
        -----------
        ripflow_signal : array-like
            Raw RIPflow signal
        auto_detect : bool
            Whether to automatically detect clipping
        threshold_percentile : float
            Percentile for clipping detection
            
        Returns:
        --------
        processed_signal : array
            Preprocessed (de-clipped if needed) signal
        clipped_regions : boolean array
            Mask showing which regions were clipped
        """
        signal = np.array(ripflow_signal)
        
        if auto_detect:
            clipped_mask, threshold = self.detect_clipping(signal, threshold_percentile)
            
            if np.any(clipped_mask):
                # Apply DORIS de-clipping
                processed_signal, clipped_mask = self.declip_signal(signal, clipped_mask)
                return processed_signal
            else:
                return signal
        else:
            return signal
        

class BreathToBreathDetrend:
    def __init__(self, sampling_rate=100):
        """
        RIPflow preprocessing including polarity correction and baseline removal
        
        Parameters:
        -----------
        sampling_rate : float
            Sampling rate in Hz
        """
        self.fs = sampling_rate
        self.min_breath_duration = 1.0  # seconds
        self.max_breath_duration = 15.0  # seconds
        self.zero_flow_threshold = 120.0  # seconds
        self.moving_avg_breaths = 5
    
    def correct_polarity(self, signal_data):
        """
        Ensure inspiration is upward (positive direction)
        """
        signal_data = np.array(signal_data)
        
        # Remove DC component for analysis
        detrended = signal_data - np.mean(signal_data)
        
        # Find peaks (potential inspirations) and troughs (potential expirations)
        peaks, _ = signal.find_peaks(detrended, height=np.std(detrended)*0.5, 
                                   distance=int(self.fs * self.min_breath_duration))
        troughs, _ = signal.find_peaks(-detrended, height=np.std(detrended)*0.5, 
                                     distance=int(self.fs * self.min_breath_duration))
        
        # Calculate breath characteristics for both orientations
        if len(peaks) > 0 and len(troughs) > 0:
            # Current orientation: peaks as inspiration
            peak_prominences = signal.peak_prominences(detrended, peaks)[0]
            trough_prominences = signal.peak_prominences(-detrended, troughs)[0]
            
            # Measure breath regularity (coefficient of variation)
            if len(peaks) > 1:
                peak_intervals = np.diff(peaks) / self.fs
                peak_cv = np.std(peak_intervals) / np.mean(peak_intervals)
            else:
                peak_cv = float('inf')
                
            if len(troughs) > 1:
                trough_intervals = np.diff(troughs) / self.fs
                trough_cv = np.std(trough_intervals) / np.mean(trough_intervals)
            else:
                trough_cv = float('inf')
            
            # Decision criteria:
            # 1. Higher average prominence
            # 2. More regular intervals (lower CV)
            # 3. More physiological breathing rate (8-30 breaths/min)
            
            avg_peak_prominence = np.mean(peak_prominences)
            avg_trough_prominence = np.mean(trough_prominences)
            
            # Calculate breathing rates
            if len(peaks) > 1:
                peak_rate = 60 / np.mean(peak_intervals)
            else:
                peak_rate = 0
                
            if len(troughs) > 1:
                trough_rate = 60 / np.mean(trough_intervals)
            else:
                trough_rate = 0
            
            # Score each orientation
            peak_score = 0
            trough_score = 0
            
            # Prominence scoring
            if avg_peak_prominence > avg_trough_prominence:
                peak_score += 1
            else:
                trough_score += 1
            
            # Regularity scoring
            if peak_cv < trough_cv:
                peak_score += 1
            else:
                trough_score += 1
            
            # Physiological rate scoring (8-30 breaths/min)
            if 8 <= peak_rate <= 30:
                peak_score += 1
            if 8 <= trough_rate <= 30:
                trough_score += 1
            
            # Decision
            if trough_score > peak_score: # Troughs are better inspirations
                is_inverted = True
            else: # Peaks are inspirations, keep as is
                is_inverted = False
        else: # Fallback: assume signal is correct
            is_inverted = False
        
        return is_inverted
    
    def detect_breath_changepoints(self, signal_data):
        """
        Detect breath changepoints (zero crossings and inspiration starts)
        
        Parameters:
        -----------
        signal_data : array
            Preprocessed signal with correct polarity
            
        Returns:
        --------
        inspiration_starts : array
            Indices of inspiration start points
        zero_crossings : array
            All zero crossing points
        """
        # Remove trend for better zero crossing detection
        detrended = signal.detrend(signal_data, type='linear')
        
        # Find zero crossings
        zero_crossings = np.where(np.diff(np.signbit(detrended)))[0]
        
        if len(zero_crossings) == 0:
            return np.array([]), np.array([])
        
        # Classify crossings as inspiration starts or expiration starts
        inspiration_starts = []
        
        for i, crossing in enumerate(zero_crossings):
            # Look at the direction of crossing
            if crossing < len(detrended) - 1:
                # Check if signal goes from neg to pos (inspiration start)
                before = detrended[crossing]
                after = detrended[crossing + 1]
                
                if before <= 0 and after > 0:
                    inspiration_starts.append(crossing)
        
        inspiration_starts = np.array(inspiration_starts)
        
        # Filter based on breath duration constraints
        if len(inspiration_starts) > 1:
            breath_durations = np.diff(inspiration_starts) / self.fs
            valid_breaths = np.where(
                (breath_durations >= self.min_breath_duration) & 
                (breath_durations <= self.max_breath_duration)
            )[0]
            
            # Keep inspiration starts that lead to valid breaths
            if len(valid_breaths) > 0:
                inspiration_starts = inspiration_starts[np.concatenate([[0], valid_breaths + 1])]
        
        return inspiration_starts
    
    def refine_inspiration_timing(self, signal_data, initial_inspirations):
        """
        Refine inspiration start timing based on inspiration time characteristics
        
        Parameters:
        -----------
        signal_data : array
            Signal data
        initial_inspirations : array
            Initial inspiration start indices
            
        Returns:
        --------
        refined_inspirations : array
            Refined inspiration start indices
        """
        if len(initial_inspirations) < 2:
            return initial_inspirations
        
        refined_inspirations = []
        
        for i in range(len(initial_inspirations)):
            start_idx = initial_inspirations[i]
            
            # Define search window around the initial detection
            search_window = int(0.2 * self.fs)  # ±200ms
            start_search = max(0, start_idx - search_window)
            end_search = min(len(signal_data), start_idx + search_window)
            
            # Find the actual minimum in the search window (true inspiration start)
            search_segment = signal_data[start_search:end_search]
            local_min_idx = np.argmin(search_segment)
            refined_start = start_search + local_min_idx
            
            refined_inspirations.append(refined_start)
        
        return np.array(refined_inspirations)
    
    def segment_breaths(self, signal_data, inspiration_starts):
        """
        Segment individual breaths from inspiration start to inspiration start
        
        Parameters:
        -----------
        signal_data : array
            Signal data
        inspiration_starts : array
            Inspiration start indices
            
        Returns:
        --------
        breath_segments : list
            List of dictionaries containing breath information
        """
        breath_segments = []
        
        for i in range(len(inspiration_starts) - 1):
            start_idx = inspiration_starts[i]
            end_idx = inspiration_starts[i + 1]
            
            breath_data = signal_data[start_idx:end_idx]
            breath_duration = (end_idx - start_idx) / self.fs
            
            # Calculate breath characteristics
            breath_max = np.max(breath_data)
            breath_min = np.min(breath_data)
            breath_amplitude = breath_max - breath_min
            
            breath_info = {
                'start_idx': start_idx,
                'end_idx': end_idx,
                'duration': breath_duration,
                'data': breath_data,
                'amplitude': breath_amplitude,
                'max_value': breath_max,
                'min_value': breath_min,
                'baseline_start': breath_data[0],
                'baseline_end': breath_data[-1]
            }
            
            breath_segments.append(breath_info)
        
        return breath_segments
    
    def calculate_moving_baseline(self, breath_segments):
        """
        Calculate moving average baseline using 5 prior breaths
        
        Parameters:
        -----------
        breath_segments : list
            List of breath segment dictionaries
            
        Returns:
        --------
        baselines : array
            Baseline values for each breath
        """
        baselines = np.zeros(len(breath_segments))
        
        for i in range(len(breath_segments)):
            # Use up to 5 prior breaths for moving average
            start_breath = max(0, i - self.moving_avg_breaths)
            
            if i == 0:
                # For first breath, use its own start value
                baselines[i] = breath_segments[i]['baseline_start']
            else:
                # Calculate moving average of baseline values from prior breaths
                prior_baselines = []
                for j in range(start_breath, i):
                    # Use the minimum value of each breath as baseline reference
                    prior_baselines.append(breath_segments[j]['min_value'])
                
                if len(prior_baselines) > 0:
                    baselines[i] = np.mean(prior_baselines)
                else:
                    baselines[i] = breath_segments[i]['baseline_start']
        
        return baselines
    
    def apply_baseline_correction(self, signal_data, breath_segments, baselines):
        """
        Apply baseline correction to the signal
        
        Parameters:
        -----------
        signal_data : array
            Original signal data
        breath_segments : list
            Breath segment information
        baselines : array
            Baseline values for each breath
            
        Returns:
        --------
        corrected_signal : array
            Baseline-corrected signal
        """
        corrected_signal = signal_data.copy()
        
        for i, breath in enumerate(breath_segments):
            start_idx = breath['start_idx']
            end_idx = breath['end_idx']
            
            # Subtract the baseline from this breath segment
            corrected_signal[start_idx:end_idx] -= baselines[i]
        
        return corrected_signal
    
    def preprocess_baseline_removal(self, signal_data):
        """
        Complete baseline removal preprocessing pipeline
        
        Parameters:
        -----------
        signal_data : array-like
            Input RIPflow signal
        plot_results : bool
            Whether to plot intermediate results
            
        Returns:
        --------
        processed_signal : array
            Baseline-corrected signal
        breath_info : dict
            Dictionary containing breath segmentation information
        valid_mask : array
            Boolean mask indicating valid regions (excluding long zero-flow periods)
        """
        signal_data = np.array(signal_data)
        
        # Step 1: Correcting signal polarity
        is_inverted = self.correct_polarity(signal_data)
        if is_inverted:
            corrected_signal = - signal_data
        else:
            corrected_signal = signal_data
        
        # Step 2: Detecting breath changepoints
        inspiration_starts = self.detect_breath_changepoints(corrected_signal)
        if len(inspiration_starts) < 2:
            # Insufficient breath detections. Returning original signal
            return corrected_signal, {}, np.ones_like(corrected_signal, dtype=bool)
        
        # Step 3: Refining inspiration timing
        refined_inspirations = self.refine_inspiration_timing(corrected_signal, inspiration_starts)
        
        # Step 4: Segmenting individual breaths
        breath_segments = self.segment_breaths(corrected_signal, refined_inspirations)
        if len(breath_segments) == 0:
            # Warning: No valid breath segments found. Returning original signal
            return corrected_signal, {}, np.ones_like(corrected_signal, dtype=bool)

        # Step 5: Calculating moving baseline
        baselines = self.calculate_moving_baseline(breath_segments)
        
        # Step 6: Applying baseline correction
        corrected_signal = self.apply_baseline_correction(corrected_signal, breath_segments, baselines)
        
        return corrected_signal
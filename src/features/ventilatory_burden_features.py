import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
import scipy.io as sio
import scipy.sparse as sp
from scipy.signal import butter, filtfilt, savgol_filter, find_peaks, peak_prominences, savgol_filter
from scipy.optimize import minimize
from scipy.stats import mode
from scipy.fft import fft, fftfreq

def extract_vb(row, tmp_dir_sub, resp_data, sleep_stages, verbose):
    dataset = row['dataset']
    psg_id = f"sub-{row['sub_id']}_ses-{row['session']}"
    fs = float(row['sfreq_global'])
    if not resp_data:
        if verbose:
            print(f"{psg_id}: No respiratory data found → skipping")
        return None
    
    good_signal = False
    
    if "NASAL_PRESSURE" in resp_data.keys():
        resp_signal = resp_data["NASAL_PRESSURE"]['signal']
        resp_signal, bool_change_unit = check_resp_units(resp_signal, psg_id, verbose)
        if bool_change_unit and verbose:
            print(f"[INFO] {psg_id} - 'NASAL_PRES': Unit change factor 10^6")

        resp_signal = airflow_preprocess(resp_signal, fs, dataset)

        # Sanity Check 
        if sanity_check(resp_signal, fs, psg_id):
            good_signal = True


    if ("ABDOMINAL" in resp_data.keys()) and ("THORACIC" in resp_data.keys()) and not good_signal:
        abdo = resp_data["ABDOMINAL"]['signal'] # already cut to sleep only
        thor = resp_data["THORACIC"]['signal'] # already cut to sleep only
        #fs_abd  = float(resp_data["ABDOMINAL"]["sfreq_signal"])
        #fs_thor = float(resp_data["THORACIC"]["sfreq_signal"])
        # Require same fs; if not, stop
        # if not np.isclose(fs_thor, fs_abd):
        #     if verbose:
        #         print(f"{psg_id}: THORACIC fs ({fs_thor}) != ABDOMINAL fs ({fs_abd}) -> cannot derive RIPFlow")
        #     return None
        
        # Check unit (MNE)
        abdo, bool_change_unit = check_resp_units(abdo, psg_id, verbose)
        if bool_change_unit and verbose:
            print(f"[INFO] {psg_id} - 'ABDOMINAL': Unit change factor 10^6")
        thor, bool_change_unit = check_resp_units(thor, psg_id, verbose)
        if bool_change_unit and verbose:
            print(f"[INFO] {psg_id} - 'THORACIC': Unit change factor 10^6")
        
        #sfreq_resp = float(resp_data["THORACIC"]['sfreq_signal'])
        
        # Deducing RIPflow
        resp_signal = derive_ripflow(thor, abdo,fs)
        resp_signal = airflow_preprocess(resp_signal, fs, dataset)

        # Sanity Check 
        results_check = sanity_check(resp_signal, fs, psg_id)
        if results_check['passed']:
            good_signal = True
 
    # Check that we have a good airflow signal to compute VB
    if not good_signal:
        if verbose: 
            print(f"{psg_id}: Any Resp signal pass signal_check !")
        #return None
    
    # Geat df_breath  
    breath_mat = tmp_dir_sub / f"breath_{psg_id}.mat"
    df_breath = get_breath_array(breath_mat, resp_signal, fs, verbose)
    if df_breath.empty:
        if verbose:
            print(f"[WARNING] {psg_id}: No breaths detected → skipping VB extraction")
        return None

    # # Compute VB
    # results = {}
    # if len(df_breath) >= 5:
    #     amp = df_breath['normalized_amplitude']
    #     amp = np.clip(amp, 0, 200)
    #     bins = np.arange(0, 205, 5)
    #     hist, _ = np.histogram(amp, bins=bins)
    #     hist_percentage = (hist/len(amp)) * 100
    #     results['VB'] = np.sum(hist_percentage[:10]) # <= 50% 
    # else:
    #     results['VB'] = np.nan

    # Sleep stages need to match sfreq_resp
    sleep_stage_per_breath = []
    for idx, row in df_breath.iterrows():
        start_idx = int(row['insp_onset'])
        stop_idx = int(row['exp_offset']) + 1
        stages_slice = sleep_stages[start_idx:stop_idx]
        valid_stages = stages_slice[~np.isnan(stages_slice)]
        
        if len(valid_stages) == 0:
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

    print(f"{psg_id} VB = {results['vb@WN']}")
    return results


# Bandpass filter Ankit values [0.01 - 2]
def bandpass_filter(signal, fs, lowcut=0.01, highcut=2, order=3):
    nyquist = fs / 2
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal


def derive_ripflow(thor, abdo, fs):
    # Z score use median for robustness
    z_abdo = (abdo - np.nanmean(abdo)) / np.nanstd(abdo)
    z_thor = (thor - np.nanmean(thor)) / np.nanstd(thor)

    # Bandpass filter Ankit values [0.01 - 2]
    filtered_abdo = bandpass_filter(z_abdo, fs)
    filtered_thor = bandpass_filter(z_thor, fs)

    # Sum signal and amplify (Ankit values)
    volume = filtered_abdo + filtered_thor
    volume = 10 * volume

    # Derive RIPflow using savgol_filter 
    ripflow = savgol_filter(volume, 
                            window_length = 201, 
                            polyorder = 3, 
                            deriv=1, 
                            delta=1.0 / fs)

    # Square Signed RIPflow
    ripflow_sq = np.sign(ripflow) * (ripflow**2)
    airflow = ripflow_sq
    return airflow


def check_resp_units(full_resp, psg_id, verbose, threshold_low=0.01, threshold_high=10.0):
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
        if verbose: 
            print(f"[INFO] {psg_id}: signal amplitude very low (ptp={ptp:.5f}), scaling by 1e6")
        return full_resp * 1e6, True
    elif ptp > threshold_high:
        if verbose:
            print(f"[INFO] {psg_id}: signal amplitude very high (ptp={ptp:.2f}), check units")
        return full_resp, False
    else:
        # likely correct units
        return full_resp, False



# Reverse signal if necessary (inspiration upward)    
def correct_polarity(signal, fs, dataset, min_breath_duration=1.0):
    """
    Ensure inspiration is upward (positive direction)
    """
    # SHHS always true
    if "shhs" in dataset:
        is_inverted = True
        return is_inverted
    
    is_inverted = False
    signal = np.array(signal)

    # Remove DC component for analysis
    detrended = signal - np.mean(signal)
    
    # Find peaks (potential inspirations) and troughs (potential expirations)
    peaks, _ = find_peaks(detrended, height=np.std(detrended)*0.5, 
                            distance=int(fs * min_breath_duration))
    troughs, _ = find_peaks(-detrended, height=np.std(detrended)*0.5, 
                            distance=int(fs * min_breath_duration))
    
    # Calculate breath characteristics for both orientations
    if len(peaks) > 0 and len(troughs) > 0:
        # Current orientation: peaks as inspiration
        peak_prominences_ = peak_prominences(detrended, peaks)[0]
        trough_prominences_ = peak_prominences(-detrended, troughs)[0]
        
        # Measure breath regularity (coefficient of variation)
        if len(peaks) > 1:
            peak_intervals = np.diff(peaks) / fs
            peak_cv = np.std(peak_intervals) / np.mean(peak_intervals)
        else:
            peak_cv = float('inf')
            
        if len(troughs) > 1:
            trough_intervals = np.diff(troughs) / fs
            trough_cv = np.std(trough_intervals) / np.mean(trough_intervals)
        else:
            trough_cv = float('inf')
        
        # Decision criteria:
        # 1. Higher average prominence
        # 2. More regular intervals (lower CV)
        # 3. More physiological breathing rate (8-30 breaths/min)
        
        avg_peak_prominence = np.mean(peak_prominences_)
        avg_trough_prominence = np.mean(trough_prominences_)
        
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
        if trough_score > peak_score:
            is_inverted = True
    
    return is_inverted


def sanity_check(full_resp, sfreq_resp, psg_id, ptp_threshold=0.01, noise_threshold=0.7):
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

class DORISDeClipper:
    def __init__(self, lambda_smooth=1.0, lambda_energy=0.1):
        """
        DORIS: De-clipping using Optimization with Regularized Interpolation and Smoothness
        lambda_smooth : Regularization parameter for smoothness constraint
        lambda_energy : Regularization parameter for finite energy constraint
        """
        self.lambda_smooth = lambda_smooth
        self.lambda_energy = lambda_energy

    # Determine if signal clipped (if not then no declipped apply)
    def detect_clipping(
        self,
        signal,
        threshold_percentile=99,
        slope_percentile=10,
        min_run_length=5
    ):
        abs_sig = np.abs(signal)

        amp_thresh = np.percentile(abs_sig, threshold_percentile)
        slope = np.abs(np.diff(signal, prepend=signal[0]))
        slope_thresh = np.percentile(slope, slope_percentile)

        # candidate clipped points
        clipped_mask = (abs_sig >= amp_thresh) & (slope <= slope_thresh)

        # remove isolated points → keep only flat runs
        clipped_mask_clean = np.zeros_like(clipped_mask, dtype=bool)

        idx = np.where(clipped_mask)[0]
        if len(idx) == 0:
            return clipped_mask_clean, amp_thresh

        runs = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)
        for r in runs:
            if len(r) >= min_run_length:
                clipped_mask_clean[r] = True

        return clipped_mask_clean, amp_thresh
    
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

    def preprocess_airflow(self, airflow, threshold_percentile=99):
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
        signal = np.array(airflow)
        clipped_mask, threshold = self.detect_clipping(signal, threshold_percentile)
        
        if np.any(clipped_mask):
            #print(f"Clipping detected at threshold {threshold:.3f}")
            #print(f"Number of clipped samples: {np.sum(clipped_mask)} ({100*np.sum(clipped_mask)/len(signal):.1f}%)")
            
            # Apply DORIS de-clipping
            processed_signal, clipped_mask = self.declip_signal(signal, clipped_mask)
            return processed_signal, clipped_mask
        else:
            #print("No clipping detected. Using raw signal.")
            return signal, np.zeros_like(signal, dtype=bool)


class BreathDetrender:
    def __init__(self, signal, fs=None):
        """
        signal : 1D numpy array
        fs     : sampling frequency (optional, only for plotting in seconds)
        """
        self.signal = np.asarray(signal)
        self.fs = fs

        # Outputs (filled progressively)
        self.breath_segments = None
        self.breath_baselines = None
        self.detrended_signal = None


    # -----------------------------------------------------
    # 1. Split signal into breaths
    # -----------------------------------------------------
    def detect_inspiration_starts(self):
        threshold = 70
        max_breath_rate = 30

        # Need to reverse to find down peak
        invert_signal = -self.signal

        # Calculate signal statistics for thresholding
        min_distance_samples = int(self.fs * 60 / max_breath_rate)
        height_thresh = np.percentile(invert_signal, threshold)
        prominence_thresh = np.percentile(np.abs(np.diff(invert_signal)), threshold)
        valleys, _ = find_peaks(
            invert_signal, # finding the down peaks
            distance=min_distance_samples,   # Minimum distance between peaks
            height=height_thresh,            # Minimum peak height
            prominence=prominence_thresh,    # Minimum peak prominence
        )

        # Find zero crossings
        zero_crossings = np.where((self.signal[:-1] <= 0) & (self.signal[1:] > 0))[0] + 1
        
        # Find first zero crossing after each valley
        inspiration_starts = []
        for valley_idx in valleys:
            future_crossings = zero_crossings[zero_crossings > valley_idx]
            if len(future_crossings) > 0:
                inspiration_starts.append(future_crossings[0]) # take first zero
        inspiration_starts = np.array(inspiration_starts)

        # Build breath segments
        breath_segments = [
            (inspiration_starts[i], inspiration_starts[i + 1])
            for i in range(len(inspiration_starts) - 1)
        ]
        self.breath_segments = breath_segments

        return self.breath_segments
    
    # -----------------------------------------------------
    # 2. Compute baseline per breath
    # -----------------------------------------------------
    def compute_breath_baselines(self):
        baselines = []

        for start, end in self.breath_segments:
            # Guard against invalid segments
            if start >= end or start < 0 or end > len(self.signal):
                baselines.append(np.nan)
                continue

            seg = self.signal[start:end]

            # Guard against empty or all-NaN segments
            if seg.size == 0 or np.all(np.isnan(seg)):
                baselines.append(np.nan)
            else:
                baselines.append(np.nanmean(seg))

        self.breath_baselines = np.asarray(baselines)
        return self.breath_baselines

    # -----------------------------------------------------
    # 3. Detrend using average of previous N breaths
    # -----------------------------------------------------
    def detrend(self):
        """
        Subtracts baseline computed as the average
        of the previous n_prior breath baselines.
        """
        n_prior = 5 # look at 5 previosu breaths
        detrended = self.signal.copy()

        for i, (start, end) in enumerate(self.breath_segments):
            if i < n_prior:
                baseline = np.nanmean(self.breath_baselines[:i+1])
            else:
                baseline = np.nanmean(self.breath_baselines[i-n_prior:i])

            detrended[start:end] -= baseline
        self.detrended_signal = detrended
        return detrended
    
    # Apply the three steps 
    def preprocess_airflow(self):
        self.detect_inspiration_starts()
        self.compute_breath_baselines()
        self.detrend()
        return self.detrended_signal
    

def airflow_preprocess(airflow, fs, dataset):
    if correct_polarity(airflow, fs, dataset, min_breath_duration=1.0):
        airflow = -airflow

    doris = DORISDeClipper(lambda_smooth=1.0, lambda_energy=0.01)
    airflow_declipped, clipped_mask = doris.preprocess_airflow(
        airflow,
        threshold_percentile=99.0  
    )

    breath_detrend = BreathDetrender(airflow_declipped, fs=fs)
    airflow_detrended = breath_detrend.preprocess_airflow()

    return airflow_detrended
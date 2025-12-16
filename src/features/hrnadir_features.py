import numpy as np
import pandas as pd
import neurokit2 as nk

def extract_hrnadir(config, psg_id, ecg_data, sleep_stages):
    """
    Compute HRnadir (minimum heart rate) from ECG and sleep stage data.

    Parameters
    ----------
    config : object
        Configuration object with attributes:
        - analysis.max_rri_sec : float, max allowed RR interval in seconds
        - run.verbose : bool, whether to print warnings
    ecg_data : dict
        Dictionary containing:
        - "clean_rpeaks": array of R-peak indices
        - "sfreq_signal": sampling frequency of ECG
    sleep_stages : array-like
        Sleep stage labels per epoch (0: W, 1: N1, 2: N2, 3: N3, 4: REM)
    epoch_length_sec : float
        Duration of one epoch in seconds

    Returns
    -------
    dict or None
        Dictionary with keys:
        - 'hrnadir' : minimum HR in bpm
        - 'hrnadir_time_hours' : time of HRnadir in hours
        - 'hrnadir_sleep_stages' : sleep stage text at HRnadir
        Returns None if computation fails.
    """

    if ecg_data is None:
        print(f"[WARNING] {psg_id}:  ecg_data is None, skipping HRnadir computation.")
        return None

    rpeaks = ecg_data.get("clean_rpeaks")
    sfreq_ecg = ecg_data.get("sfreq_signal")
    if rpeaks is None or sfreq_ecg is None or len(rpeaks) == 0:
        print(f"[WARNING] {psg_id}:  rpeaks is None, skipping HRnadir computation.")
        return None

    # Compute HR trace in bpm
    hr = nk.ecg_rate(
        rpeaks,
        sampling_rate=sfreq_ecg,
        desired_length=len(sleep_stages)
    )

    # Define physiological HR bounds (from min/max RRI)
    hr_before = np.isnan(hr).copy()
    hr_lower_bound = 60 / float(config.analysis.max_rri_sec)  # bpm lower bound
    hr_upper_bound = 60 / float(config.analysis.min_rri_sec)  # bpm upper bound
    hr[(hr < hr_lower_bound) | (hr > hr_upper_bound)] = np.nan
    hr_after = np.isnan(hr)
    new_nans = np.sum(hr_after & ~hr_before)

    if config.run.verbose: # Should be 0 because already done in clean rpeaks
        n_nans = np.isnan(hr).sum()
        print(f"[WARNING] {psg_id}: Outliers in HR trace: {new_nans} / {len(hr)} "
              f"({new_nans / len(hr) * 100:.2f}%)")

    # Smooth HR with 10-second rolling median (ignores NaNs)
    hr = pd.Series(hr).rolling(window=int(sfreq_ecg*10), center=True, min_periods=1).median().values

    # Mask HR values outside valid sleep stage
    hr[np.isnan(sleep_stages)] = np.nan

    if np.all(np.isnan(hr)):
        print(f"[WARNING] {psg_id}:  All HR values are NaN, skipping HRnadir computation.")
        return None

    # Index of minimum HR
    idx = np.nanargmin(hr)

    # Map numeric stage to text
    stage_num2txt = {0:'W', 1:'N1', 2:'N2', 3:'N3', 4:'REM'}
    stage_text = stage_num2txt.get(int(sleep_stages[idx]), "Unknown")

    return {
        'hrnadir': hr[idx],
        'hrnadir_time_hours': idx / sfreq_ecg / 3600,
        'hrnadir_sleep_stages': stage_text
    }
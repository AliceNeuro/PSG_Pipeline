import mne
import numpy as np
import neurokit2 as nk
from scipy.signal import peak_prominences
from itertools import groupby

 
def ecg_preprocessing(full_ecg, sfreq_ecg, full_sleep_stages):

    flip = determine_flip_ecg(full_ecg, sfreq_ecg)
    if flip:
        full_ecg = -full_ecg

    full_ecg = filter_ecg(full_ecg, sfreq_ecg)

    ecg = take_sleep_part(full_ecg, sfreq_ecg, full_sleep_stages)

    return full_ecg, ecg, sfreq_ecg


def determine_flip_ecg(ecg, sfreq_ecg, line_freq=60):
    # NaN/Inf mask
    nan_mask = np.isnan(ecg) | np.isinf(ecg) | (np.abs(ecg) > 5000)

    cc = 0
    maxll = 0
    start = -1
    for k, l in groupby(nan_mask):
        ll = len(list(l))
        if (not k) and ecg[cc:cc + ll].std() > 1e-5 and ll > maxll:
            maxll = ll
            start = cc
        cc += ll

    if start < 0:
        return None
    
    ecg_seg = ecg[start:start + maxll]
    if len(ecg_seg) < sfreq_ecg * 600:
        return None
    if len(ecg_seg) > sfreq_ecg * 7200:
        start = int(len(ecg_seg) // 2 - sfreq_ecg * 3600)
        end = int(len(ecg_seg) // 2 + sfreq_ecg * 3600)
        ecg_seg = ecg_seg[start:end]

    # Notch Filter
    if line_freq < sfreq_ecg / 2:
        ecg_seg = mne.filter.notch_filter(ecg_seg.astype(np.float64), sfreq_ecg, 60, verbose=False)
    
    # Band Pass Filter
    lowcut = 3  # remove slow drift only
    highcut = min(70, sfreq_ecg / 2 - 1) 
    ecg_seg = mne.filter.filter_data(ecg_seg.astype(np.float64), sfreq_ecg, lowcut, highcut, verbose=False)

    # Find peaks for both signals 
    try:
        rpeaks1 = nk.ecg_peaks(ecg_seg, sampling_rate=sfreq_ecg)[1]['ECG_R_Peaks']
        rpeaks2 = nk.ecg_peaks(-ecg_seg, sampling_rate=sfreq_ecg)[1]['ECG_R_Peaks']
    except Exception as ee:
        print(str(ee))
        return None
    peakness1 = np.mean(peak_prominences(ecg_seg, rpeaks1)[0])
    peakness2 = np.mean(peak_prominences(-ecg_seg, rpeaks2)[0])
    hr = len(rpeaks2 if peakness1 < peakness2 else rpeaks1) / (len(ecg_seg) / sfreq_ecg / 60)
    if 30 < hr < 120:
        return peakness1 < peakness2
    else:
        return None


def filter_ecg(ecg, sfreq_ecg):
    ecg = ecg.astype(np.float64)
    if sfreq_ecg > 120:
        ecg = mne.filter.notch_filter(ecg, sfreq_ecg, 60, verbose=False)
    lowcut = 3 #3 # remove slow drift only
    highcut = min(70, sfreq_ecg / 2 - 1) 
    ecg= mne.filter.filter_data(ecg, sfreq_ecg, lowcut, highcut, verbose=False)
    return ecg


def take_sleep_part(full_ecg, sfreq_ecg, full_sleep_stages):
    # Keep only sleep (same as process_sleep_stages)
    sleep_ids = np.where(np.isin(full_sleep_stages, [1, 2, 3, 4]))[0]
    start_index = sleep_ids[0]
    end_index = sleep_ids[-1] + 1 # include the last index

    # Cut sleep stages and ecg
    ecg = full_ecg[int(start_index):int(end_index)]

    return ecg
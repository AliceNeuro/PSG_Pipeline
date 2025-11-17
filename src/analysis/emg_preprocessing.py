import h5py
import numpy as np
from pathlib import Path

def emg_preprocessing(h5_path, full_sleep_stages, tmp_dir_sub, rpeaks_detection_method):
    emg_type = None
    full_emg, sfreq_emg = read_emg(h5_path, emg_type)

    full_emg = filter_emg(full_emg, sfreq_emg)

    return full_emg, emg, sfreq_emg

def read_emg(h5_path, emg_type):
    if not Path(h5_path).exists():
        raise FileNotFoundError(f"File not found: {h5_path}")
    
    emg_key = f'signals/EMG/EMG_{emg_type}'

    with h5py.File(h5_path, 'r') as f:
        if emg_key not in f:
            raise KeyError(f"{emg_key} not found in file {h5_path}")
        
        full_emg = f[emg_key][:]
        sfreq_emg = f[emg_key].attrs['fs']
    
    return full_emg, sfreq_emg


def filter_emg(emg, sfreq_emg):
    emg = emg.astype(np.float64)
    '''
    if sfreq_emg > 120:
        emg = mne.filter.notch_filter(emg, sfreq_emg, 60, verbose=False)
    emg = mne.filter.filter_data(emg, sfreq_emg, 3, 70 if sfreq_emg > 140 else None, verbose=False)
    '''
    return emg


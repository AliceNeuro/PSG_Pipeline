import h5py
import numpy as np
from pathlib import Path
from analysis.peak_detection import peak_detection
from analysis.ecg_preprocessing import ecg_preprocessing
from analysis.emg_preprocessing import emg_preprocessing

def process_signals(config, row, full_sleep_stages, tmp_dir_sub):
    psg_id = f"sub-{row['sub_id']}_ses-{row['session']}"
    SIGNAL_ANALYSIS_FUNCTIONS = {
        "ecg": lambda c, h5, f, tmp: ecg_analysis(c, h5, f, tmp),
        "emg": lambda c, h5, f, tmp: emg_analysis(c, h5, f, tmp),
        "spo2": lambda c, h5, f, tmp: spo2_analysis(c, h5, f, tmp),
        "resp": lambda c, h5, f, tmp: resp_analysis(c, h5, f, tmp),
    }

    processed_signals = {}
    signals_to_analyze = get_signals_to_analyze(config)

    for signal in signals_to_analyze:
        func = SIGNAL_ANALYSIS_FUNCTIONS.get(signal)
        if func:
            if config.run.verbose:
                print(f"[INFO]: Processing Signal: {signal}")
            result = func(config,
                          row['h5_path'],
                          full_sleep_stages, 
                          tmp_dir_sub)
            if isinstance(result, dict):
                for signal_name, signal_results in result.items():
                    processed_signals[signal_name] = signal_results
            else:
                print(f"[WARNING] {psg_id}: Analysis for {signal} did not return a dict")
        else:
            print(f"[WARNING] {psg_id}:  No analysis function defined for signal: {signal}")
    return processed_signals
  
def get_signals_to_analyze(config):
    """
    Determine which signals need to be analyzed based on the config.
    """
    if config.features.extract_all:
        return ['ecg', 'emg', 'spo2', 'airflow']

    selected = config.features.selected or []
    signals_to_analyze = set()

    if "hrv" in selected:
        signals_to_analyze.add("ecg")
    if "cpc" in selected: 
        signals_to_analyze.add("ecg")
    if "hrnadir" in selected: 
        signals_to_analyze.add("ecg")
    if "hb" in selected: 
        signals_to_analyze.add("spo2")
    if "vb" in selected: 
        signals_to_analyze.add("resp")

    return sorted(signals_to_analyze)

def ecg_analysis(config, h5_path, full_sleep_stages, tmp_dir_sub):
    """
    Processes ECG signals in an H5 file. Preprocesses all signals, 
    runs peak detection in order of preference, and returns results.
    """
    if not Path(h5_path).exists():
        raise FileNotFoundError(f"File not found: {h5_path}")

    results = {}

    # --- Step 1: preprocess all ECGs ---
    with h5py.File(h5_path, 'r') as f:
        ecg_data = {}
        for ecg_signal in f['signals/ECG']:
            full_ecg = f['signals/ECG'][ecg_signal][:]
            sfreq_ecg = f['signals/ECG'][ecg_signal].attrs.get('fs', None)

            full_ecg_processed, ecg_sleep, sfreq_ecg = ecg_preprocessing(
                full_ecg,
                sfreq_ecg,
                full_sleep_stages,
            )

            ecg_data[ecg_signal] = {
                "full_signal": full_ecg_processed,
                "signal": ecg_sleep,
                "sfreq_signal": sfreq_ecg,
                "clean_rpeaks": None, 
                "clean_rpeaks_mask": None, 
            }

        # Add substracted ECG if 2 of them
        if 'ECG_L' in f['signals/ECG'] and 'ECG_R' in f['signals/ECG']:
            # Load the two channels
            ecg_l = f['signals/ECG']['ECG_L'][:]
            ecg_r = f['signals/ECG']['ECG_R'][:]
            sfreq_ecg = f['signals/ECG']['ECG_L'].attrs.get('fs', None)

            # Subtract ECG_R from ECG_L
            full_ecg = ecg_l - ecg_r

            # Run preprocessing
            full_ecg_processed, ecg_sleep, sfreq_ecg = ecg_preprocessing(
                full_ecg,
                sfreq_ecg,
                full_sleep_stages,
            )

            # Store in dictionary
            ecg_data['ECG'] = {
                "full_signal": full_ecg_processed,
                "signal": ecg_sleep,
                "sfreq_signal": sfreq_ecg,
                "clean_rpeaks": None,
                "clean_rpeaks_mask": None,
            }

    # --- Step 2: Peak Detection ---
    clean_rpeaks, clean_rpeaks_mask, chosen_ecg = peak_detection(config, ecg_data, tmp_dir_sub)

    # --- Step 3: Assign Results ---
    for signal_name, signal_data in ecg_data.items():
        results[signal_name] = signal_data
        if signal_name == chosen_ecg:
            results[signal_name]["clean_rpeaks"] = clean_rpeaks
            results[signal_name]["clean_rpeaks_mask"] = clean_rpeaks_mask
        else:
            results[signal_name]["clean_rpeaks"] = None
            results[signal_name]["clean_rpeaks_mask"] = None

    return results

def emg_analysis(config, h5_path, full_sleep_stages, tmp_dir_sub):
    """
    Processes ECG data and performs peak detection. Returns a dictionary with
    signal and metadata.
    """
    full_emg, emg, sfreq_emg = emg_preprocessing(
        full_sleep_stages, 
        h5_path,
        tmp_dir_sub)                      
                                                    
    return {"EMG": {
                "full_signal": full_emg,
                "signal": emg,
                "sfreq_signal": sfreq_emg,
            }}

def eeg_analysis(config, row, sleep_stages, df_events):
    return None

def spo2_analysis(config, h5_path, full_sleep_stages, tmp_dir_sub):
    key = 'signals/SPO2/SPO2'
    with h5py.File(h5_path, 'r') as f:
        if key not in f:
            raise KeyError(f"{key} not found in file {h5_path}")
        full_spo2 = f[key][:]
        sfreq_spo2 = f[key].attrs['fs']
    
    spo2 = take_sleep_part(full_spo2, full_sleep_stages)

    return {"SPO2": {
                "full_signal": full_spo2,
                "signal": spo2,
                "sfreq_signal": sfreq_spo2,
            }}

def resp_analysis(config, h5_path, full_sleep_stages, tmp_dir_sub):
    keys = {
        "NASAL_PRESSURE": "signals/RESP/RESP_NASAL_PRESSURE",
        "THERM": "signals/RESP/RESP_THERM"
    }
    results = {}

    with h5py.File(h5_path, 'r') as f:
        for name, key in keys.items():
            if key in f:
                full_signal = f[key][:]
                sfreq_signal = f[key].attrs['fs']
                signal = take_sleep_part(full_signal, full_sleep_stages)

                results[name] = {
                    "full_signal": full_signal,
                    "signal": signal,
                    "sfreq_signal": sfreq_signal,
                }

    if not results:
        psg_id = tmp_dir_sub.name
        print(f"[WARNING] {psg_id}: Neither NASAL_PRESSURE nor THERM channels found in {h5_path}")
        return {}

    return results

def take_sleep_part(full_signal, full_sleep_stages):
    # Keep only sleep (same as process_sleep_stages)
    sleep_ids = np.where(np.isin(full_sleep_stages, [1, 2, 3, 4]))[0]
    start_index = sleep_ids[0]
    end_index = sleep_ids[-1] + 1 # include the last index
    signal = full_signal[int(start_index):int(end_index)]
    return signal
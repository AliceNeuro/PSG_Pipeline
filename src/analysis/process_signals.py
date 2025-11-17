import h5py
from analysis.peak_detection import peak_detection
from analysis.ecg_preprocessing import ecg_preprocessing
from analysis.emg_preprocessing import emg_preprocessing

def process_signals(config, row, full_sleep_stages, tmp_dir_sub):

    SIGNAL_ANALYSIS_FUNCTIONS = {
        "ecg": lambda c, h5, f, tmp: ecg_analysis(c, h5, f, tmp),
        "emg": lambda c, h5, f, tmp: emg_analysis(c, h5, f, tmp),
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
                print(f"[WARNING] Sub {row['sub_id']}: Analysis for {signal} did not return a dict")
        else:
            print(f"[WARNING] Sub {row['sub_id']}:  No analysis function defined for signal: {signal}")
    return processed_signals
  
def get_signals_to_analyze(config):
    """
    Determine which signals need to be analyzed based on the config.
    """
    if config.features.extract_all:
        return ['ecg', 'emg']

    selected = config.features.selected or []
    signals_to_analyze = set()

    if "hrv" in selected:
        signals_to_analyze.add("ecg")
    if "cpc" in selected: 
        signals_to_analyze.add("ecg")
    if "hrnadir" in selected: 
        signals_to_analyze.add("ecg")

    return sorted(signals_to_analyze)

def ecg_analysis(config, h5_path, full_sleep_stages, tmp_dir_sub):
    """
    Processes ECG signals in an H5 file. Preprocesses all signals, 
    runs peak detection in order of preference, and returns results.
    """
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
                                                    
    return {
        "full_signal": full_emg,
        "signal": emg,
        "sfreq_signal": sfreq_emg,
    }

def eeg_analysis(config, row, sleep_stages, df_events):
    return None
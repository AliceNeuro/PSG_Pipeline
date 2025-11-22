import matplotlib.pyplot as plt
import numpy as np
import mne 
import math
import yasa
from sklearn.metrics import confusion_matrix, cohen_kappa_score, f1_score, accuracy_score, precision_score, recall_score
import warnings
from sklearn.exceptions import InconsistentVersionWarning, UndefinedMetricWarning


def yasa_compare(row, full_sleep_stages, psg_id):
    predicted_stages = get_predicted_stages(row)
    actual_stages = full_sleep_stages
    sfreq_global = row["sfreq_global"]

    predicted_stages = np.array(predicted_stages, dtype=float)
    actual_stages = np.array(actual_stages, dtype=float)
    min_len = min(len(predicted_stages), len(actual_stages))
    predicted_stages = predicted_stages[:min_len]
    actual_stages = actual_stages[:min_len]

    composite_score = calculate_metrics(actual_stages, predicted_stages)
    if composite_score < 0.60:
        plot_hypno_small(psg_id, composite_score, sfreq_global, predicted_stages, actual_stages)
    
    return composite_score


def get_predicted_stages(row):
    """
    Predict sleep stages using YASA for a single subject. YASA uses EEG, EOG, and EMG channels.
    
    Parameters:
    - edf_file_path (str): Path to the EDF file containing EEG, EOG, and EMG data.
    - age (int): Age of the subject.
    - gender (int): Gender of the subject (0: female, 1: male).
    
    Returns:
    - hypno_pred(list): A list of predicted sleep stages (as integers), corresponding to 30-second epochs.
    """
    edf_file_path = row['edf_path']
    age = row['age']
    gender = row['gender']

    raw = mne.io.read_raw_edf(edf_file_path, preload=False, verbose='Error')

    # Extract channel names for EEG, EOG, and EMG
    # The EEG selected is C4 so using opposite EOG (in this case left EOG)
    channel_labels = raw.ch_names
    emg_name = next((ch for ch in channel_labels if "CHIN" in ch.upper()), None)

    # Try first choice: C4 + E1
    eeg_name = next((ch for ch in channel_labels if "C4" in ch.upper()), None)
    if eeg_name:
        eog_name = next((ch for ch in channel_labels if "E1" in ch.upper()), None)
    else:
        # Fallback: C3 + E2
        eeg_name = next((ch for ch in channel_labels if "C3" in ch.upper()), None)
        if eeg_name:
            eog_name = next((ch for ch in channel_labels if "E2" in ch.upper()), None)

    if not eeg_name or not eog_name or not emg_name:
        print(channel_labels)
        raise ValueError(f"Missing required channels! EEG: {eeg_name}, EOG: {eog_name}, EMG: {emg_name}")
    

    # Predict sleep stages using YASA
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InconsistentVersionWarning)
        sls = yasa.SleepStaging(
            raw,
            eeg_name=eeg_name,
            eog_name=eog_name,
            emg_name=emg_name,
            metadata=dict(age=age, male=gender)
        )
        hypno_pred = sls.predict()
        
    hypno_pred = yasa.hypno_str_to_int(hypno_pred)
    hypno_pred_samples = np.repeat(hypno_pred, int(30 * row["sfreq_global"]))
    
    return hypno_pred_samples



def calculate_metrics(actual_stages, predicted_stages):
    """
    Calculate confusion matrix and performance metrics between actual and predicted stages.
    
    Parameters:
    - actual_stages (list): List of actual sleep stages (integers).
    - predicted_stages (list): List of predicted sleep stages (integers).
    
    Returns:
    - tuple: Contains confusion matrix and performance metrics (kappa, f1, accuracy, precision, recall, specificity).
    """
    with warnings.catch_warnings():
        # Ignore warnings for multi-class pos_label and undefined metrics
        warnings.simplefilter("ignore", category=UndefinedMetricWarning)
        warnings.simplefilter("ignore", category=UserWarning)

        mask = ~np.isnan(actual_stages) & ~np.isnan(predicted_stages)
        actual_stages = actual_stages[mask].astype(int)
        predicted_stages = predicted_stages[mask].astype(int)

        kappa = cohen_kappa_score(actual_stages, predicted_stages)
        f1 = f1_score(
            actual_stages, predicted_stages, average='weighted', zero_division=0)
        precision = precision_score(
            actual_stages, predicted_stages, average='macro', zero_division=0)
        recall = recall_score(
            actual_stages, predicted_stages, average='macro', zero_division=0)
        specificity = recall_score(
            actual_stages, predicted_stages, average='macro', pos_label=0)

        composite_score = (
                0.40 * kappa +
                0.30 * f1 +
                0.10 * precision +
                0.10 * recall +
                0.10 * specificity
            )
        
        return composite_score
    

def map_to_plot_order_safe(stages):
    mapping = {0:4, 1:2, 2:1, 3:0, 4:3}
    mapped = []
    for s in stages:
        if s is np.nan or (isinstance(s, float) and math.isnan(s)):
            mapped.append(np.nan)
        else:
            mapped.append(mapping[s])
    return np.array(mapped)


def plot_hypno_small(psg_id, composite_score, sfreq_global, predicted_stages, actual_stages):
    # Map stages
    actual_plot = map_to_plot_order_safe(actual_stages)
    predicted_plot = map_to_plot_order_safe(predicted_stages)

    # Time in sec 
    time_axis = np.arange(len(actual_plot)) / sfreq_global

    # Logical order from bottom to top: N3, N2, N1, R, W
    stage_labels = ['N3', 'N2', 'N1', 'R', 'W']

    # --- Single combined plot ---
    plt.figure(figsize=(15, 5))
    plt.step(time_axis , actual_plot, where='mid', color='blue', label='Actual')
    plt.step(time_axis , predicted_plot, where='mid', 
             color='orange', alpha=0.7, label='Predicted')

    plt.yticks(range(5), stage_labels)
    plt.title(f'{psg_id} score: {composite_score:.2f}')
    plt.xlabel('Time (s)')
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        f"/wynton/home/leng/alice-albrecht/projects/PSG_Pipeline/plots/hsp_mgb/{psg_id}_{composite_score:.2f}.png",
        dpi=150
    )
    plt.close() 
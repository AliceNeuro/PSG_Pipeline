
from hrvanalysis import *
import numpy as np


def extract_hrv(config, sleep_onset_time, ecg_data, windows_dict_ecg):
    if ecg_data is None or windows_dict_ecg is None:
        return None
    
    feat_names_hrv = [
        'mean_nni', 'sdnn', 'sdsd', 'nni_50', 'pnni_50', 'nni_20', 'pnni_20',
        'rmssd', 'median_nni', 'range_nni', 'cvsd', 'cvnni', 'mean_hr', 'max_hr', 'min_hr', 'std_hr',
        'lf', 'hf', 'lf_hf_ratio', 'lfnu', 'hfnu', 'total_power', 'vlf',
        'log_lf', 'log_hf', 'log_lf_hf_ratio', 'log_lfnu', 'log_hfnu', 'log_total_power', 'log_vlf',
        'sd1', 'sd2', 'ratio_sd2_sd1', 'sampen'
    ]

    sfreq_ecg = ecg_data["sfreq_signal"]
    rpeaks = ecg_data["clean_rpeaks"]

    window_size_sec = config.analysis.window_size_min * 60
    window_size_samples = int(round(window_size_sec * sfreq_ecg))
    features_by_stage = []

    for stage_key, info in windows_dict_ecg.items():
        stage_type = stage_key.split('@')[0]
        timestamps = info.get('timestamps', [])

        if not timestamps:
            feat_stage = {x: np.nan for x in feat_names_hrv}
            feat_stage['stage_type'] = stage_type
            features_by_stage.append(feat_stage)
            continue

        window_start_ids = [
            int(round((timestamp - sleep_onset_time).total_seconds() * sfreq_ecg))
            for timestamp in timestamps
        ]

        if config.run.verbose:
            print(f"Window start ids for {stage_type}: {window_start_ids}")

        hrv_feats = {x: [] for x in feat_names_hrv}
        for start in window_start_ids:
            end = start + window_size_samples
            rpeaks_window = rpeaks[(rpeaks >= start) & (rpeaks < end)]

            # Put in ms and clean rpeaks before getting features
            rri_ms_raw = np.diff(rpeaks_window) / sfreq_ecg * 1000
            feat = get_hrv_features(rri_ms_raw, sfreq_ecg)

            for x in feat_names_hrv:
                hrv_feats[x].append(feat.get(x, np.nan))
        
        feat_stage = {x: np.nanmean(hrv_feats[x]) for x in feat_names_hrv}
        feat_stage['stage_type'] = stage_type
        features_by_stage.append(feat_stage)

    return features_by_stage

def get_hrv_features(rri_ms_clean, sfreq_ecg):
    try:
        feat1 = get_time_domain_features(rri_ms_clean)
    except Exception as ee:
        feat1 = {}
    try:
        feat2 = get_frequency_domain_features(rri_ms_clean, sampling_frequency=sfreq_ecg, method='lomb')
        
        # Compute the log of frequency domain features
        feat2_log = {}
        for key, val in feat2.items():
            if np.isfinite(val) and val > 0:
                feat2_log[f"log_{key}"] = np.log(val)  
            else:
                feat2_log[f"log_{key}"] = np.nan 

    except Exception as ee:
        feat2 = {}
    try:
        feat3 = get_poincare_plot_features(rri_ms_clean)
    except Exception as ee:
        feat3 = {}
    try:
        feat4 = get_sampen(rri_ms_clean)
    except Exception as ee:
        feat4 = {}
    return feat1|feat2|feat2_log|feat3|feat4
 
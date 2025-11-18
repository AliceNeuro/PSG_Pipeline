import traceback
import inspect
from features.hrv_features import extract_hrv
from features.cpc_features import extract_cpc
from features.hrnadir_features import extract_hrnadir
from features.hypoxic_burden_features import extract_hb

def extract_features(config, row, tmp_dir_sub, sleep_stages, sleep_onset_time, processed_signals, df_events, windows_dict):
    FEATURE_REGISTRY = {
        "hrv": {
            "func": extract_hrv,
            "args": ["config", "sleep_onset_time", "ecg_data", "windows_dict_ecg"],
        },
        "cpc": {
            "func": extract_cpc,
            "args": ["config", "tmp_dir_sub", "sleep_onset_time", "ecg_data", "windows_dict_ecg"],
        },
        "hrnadir": {
            "func": extract_hrnadir,
            "args": ["config", "sub_id", "ecg_data", "sleep_stages"],
        },
        "hb": {
            "func": extract_hb,
            "args": ["row", "spo2_data", "df_events", "full_sleep_stages"],
        },
        # Add more features as needed
    }

    extracted_features = {}
    selected = config.features.selected or []
    extract_all = config.features.extract_all

    for feature_name, info in FEATURE_REGISTRY.items():
        if extract_all or feature_name in selected:
            func = info["func"]
            if config.run.verbose: 
                print(f"[INFO] Extracting feature: {feature_name}")

            try:
                sig = inspect.signature(func)
                func_args = sig.parameters.keys()
                
                # --- ECG data selection ---
                ecg_data = None
                windows_dict_ecg = None
                ecg_order = ["ECG", "ECG_L", "ECG_R"]

                for key in ecg_order:
                    if key in processed_signals and processed_signals[key] is not None:
                        if processed_signals[key].get("clean_rpeaks") is not None:
                            ecg_data = processed_signals[key]
                            if windows_dict is not None:
                                windows_dict_ecg = windows_dict.get(key, None)
                            else:
                                windows_dict_ecg = None
                            break  # stop at the first available ECG
                        
                all_data = {
                    "config": config,
                    "row": row,
                    "sub_id" : row["sub_id"],
                    "tmp_dir_sub": tmp_dir_sub,
                    "sleep_stages": sleep_stages,                    "sleep_onset_time": sleep_onset_time,
                    "ecg_data": ecg_data,
                    "eeg_data": processed_signals.get("EEG", {}),
                    "resp_data": processed_signals.get("RESP", {}),
                    "spo2_data": processed_signals.get("SPO2", {}),
                    "df_events": df_events,
                    "windows_dict_ecg": windows_dict_ecg
                }


                kwargs = {k: all_data[k] for k in func_args if k in all_data}
                result = func(**kwargs)

                # Post-process if feature returns stage-based values
                if result is not None:
                    if isinstance(result, list):
                        result = collapse_stage_features_to_dict(result)
                    if not isinstance(result, dict):
                        print(f"[WARNING] Sub {row['sub_id']}: Feature function '{feature_name}' did not return a dict.")
                    else:
                        extracted_features.update(result)
                else: 
                    print(f"[WARNING] Sub {row['sub_id']}: Feature '{feature_name}' was not computed: no valid R-peaks or clean windows found.")

            except Exception as e:
                print(f"[ERROR] Sub {row['sub_id']}: Failed to extract '{feature_name}': {e}")
                traceback.print_exc() 

    return extracted_features

def collapse_stage_features_to_dict(features_by_stage):
    """
    Convert list of feature dicts per stage into a flat dict with keys like 'sdnn@WN'.
    """
    flat = {}
    for entry in features_by_stage:
        stage = entry.get("stage_type", "unknown")
        for k, v in entry.items():
            if k == "stage_type":
                continue
            flat[f"{k}@{stage}"] = v
    return flat
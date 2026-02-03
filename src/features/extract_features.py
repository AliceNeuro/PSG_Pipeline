import traceback
import inspect
from features.hrv_features import extract_hrv
from features.cpc_features import extract_cpc
from features.hrnadir_features import extract_hrnadir
from features.hypoxic_burden_features import extract_hb
from features.arousal_burden_features import extract_ab
from features.ventilatory_burden_features import extract_vb
from features.aasm_features import extract_aasm

def extract_features(config, row, tmp_dir_sub, full_sleep_stages, sleep_stages, sleep_onset_offset_sec, sleep_onset_time, processed_signals, df_events, windows_dict):       
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
            "args": ["config", "psg_id", "ecg_data", "sleep_stages"],
        },
        "hb": {
            "func": extract_hb,
            "args": ["row", "spo2_data", "df_events", "full_sleep_stages", "verbose"],
        },
        "ab": {
            "func": extract_ab,
            "args": ["row", "df_events", "sleep_stages", "sleep_onset_offset_sec"],
        },
        "vb": {
            "func": extract_vb,
            "args": ["row", "tmp_dir_sub", "resp_data", "sleep_stages", "verbose"],
        },
        "aasm": {
            "func": extract_aasm,
            "args": ["full_sleep_stages", "sleep_stages", "df_events", "sfreq_global", "psg_id"],
        },

        # Add more features as needed
    }
    psg_id = f"sub-{row['sub_id']}_ses-{row['session']}"
    extracted_features = {}
    features_selected = config.features.selected if config.features.selected is not None else []
    extract_all = config.features.extract_all
    verbose = config.run.verbose

    for feature_name, info in FEATURE_REGISTRY.items():
        if extract_all or feature_name in features_selected:
            func = info["func"]
            if verbose: 
                print(f"[INFO] Extracting feature: {feature_name}")

            try:
                sig = inspect.signature(func)
                func_args = sig.parameters.keys()
                
                # --- ECG data selection ---
                ecg_data = None
                windows_dict_ecg = None
                if feature_name in ["hrv", "cpc", "hrnadir"]:
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
                    
                # --- RESP data selection ---
                resp_data = None
                if feature_name == "vb":
                    resp_data = {}
                    for resp_channel in ["NASAL_PRESSURE", "ABDOMINAL", "THORACIC", "THERM"]:
                        if resp_channel in processed_signals:
                            resp_data[resp_channel] = processed_signals[resp_channel]
                    if not resp_data and verbose:
                        print(f"[ERROR] {psg_id}: No correct respiratory channels.")
                    
                # --- Attribute All Data ---     
                all_data = {
                    "config": config,
                    "row": row,
                    "tmp_dir_sub": tmp_dir_sub,
                    "full_sleep_stages": full_sleep_stages,
                    "sleep_stages": sleep_stages,
                    "sleep_onset_offset_sec": sleep_onset_offset_sec,
                    "sleep_onset_time": sleep_onset_time,
                    "ecg_data": ecg_data,
                    "resp_data": resp_data,
                    "spo2_data": processed_signals.get("SPO2", {}),
                    "df_events": df_events,
                    "windows_dict_ecg": windows_dict_ecg,
                    "verbose":config.run.verbose,
                    "psg_id": psg_id
                }


                kwargs = {k: all_data[k] for k in func_args if k in all_data}
                result = func(**kwargs)

                # Post-process if feature returns stage-based values
                if result is not None:
                    if isinstance(result, list):
                        result = collapse_stage_features_to_dict(result)
                    if not isinstance(result, dict):
                        print(f"[WARNING] {psg_id}: Feature function '{feature_name}' did not return a dict.")
                    else:
                        extracted_features.update(result)
                else: 
                    print(f"[WARNING] {psg_id}: Feature '{feature_name}' was not computed.")

            except Exception as e:
                print(f"[ERROR] {psg_id}: Failed to extract '{feature_name}': {e}")
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
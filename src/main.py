import traceback
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from pathlib import Path
import shutil
from functools import partial
import math
import sys
import os
import re
import pandas as pd

# main
from config.read_config import read_config
from config.save_config import save_config
from pipeline_io.get_mastersheet import get_mastersheet
from pipeline_io.edf_to_h5 import edf_to_h5

# process_subjects
from pipeline_io.read_annot import read_annot
from analysis.process_sleep_stages import process_sleep_stages
from analysis.process_signals import process_signals
from analysis.selecting_windows import selecting_windows
from features.extract_features import extract_features
from pipeline_io.save_features import save_features, save_features_wide

def main():
    # --- Step 1: Read Config and Save a copy in the output ---
    config, config_path = read_config()
    save_config(config, config_path)

    # --- Step 2: Get mastersheet ---  
    mastersheet = get_mastersheet(config)
    mastersheet = mastersheet.sort_values("sub_id").reset_index(drop=True)

    # -----------------------------------------------
    # -------------- TO RUN SUBSETS -----------------
    # -----------------------------------------------

    # For HSP datasets take 3h recording minimum 
    # print(f"{len(mastersheet[mastersheet['duration_sec'] < 3*3600])} subjects with less than 3h of recording")
    # print(mastersheet[mastersheet['duration_sec'] < 3*3600])
    # mastersheet = mastersheet[mastersheet['duration_sec'] >= 3*3600]

    ### Start additional code to run only selected subs ### 
    # already_computed = []
    # for f in os.listdir(config.paths.extracted_features):
    #     if f.endswith("_wide.csv"):
    #         match = f.split("sub-")[1].split("_")[0]
    #         if match:
    #             already_computed.append(match)
    
    # mastersheet = mastersheet[~mastersheet["sub_id"].isin(already_computed)]
    # selected_subs = mastersheet["sub_id"].tolist()
    # print(f"{len(selected_subs)} selected subjects: {selected_subs}")

    # path_log = "/wynton/home/leng/alice-albrecht/PSG_Pipeline/log/mgb_all.o1060548"

    # selected_subjects_sessions = []
    # with open(path_log, "r") as f:
    #     for line in f:
    #         if "Trimming extra sleep_stages" in line:
    #             psg_id = line.split(':')[0].split('sub-')[1]
    #             sub_id = psg_id.split('_')[0]
    #             session = psg_id.split('ses-')[-1]
    #             selected_subjects_sessions.append((sub_id, int(session)))
    
    selected_subjects = [
        ("pi4519", 1),
        ("po6919", 1),
        ("sd8616", 1),
        ]
    mastersheet = mastersheet[mastersheet.apply(lambda row: (row["sub_id"], row["session"]) in selected_subjects, axis=1)]
    print(len(selected_subjects), "selected subjects/sessions:", mastersheet[["sub_id", "session"]].values.tolist())
    ### End additional code to run only selected subs ### 

    # mastersheet = mastersheet[ ~(mastersheet['annot_path'].isna() & mastersheet['sleep_stage_path'].isna())] # remove the PSG that have any event or sleep annotations
    # mastersheet = mastersheet[
    #     mastersheet['annot_path'].str.lower().str.contains("psg_annot", na=False)
    # ]
    # mastersheet = mastersheet[
    #     mastersheet['annot_path'].str.lower().str.contains("xltek", na=False)
    # ]
    #mastersheet = mastersheet[:8]
    # Define subjects/sessions to exclude

    # Exclude the ones wihtout annot_path
    # print(len(mastersheet[mastersheet['annot_path'].isna()])," PSG without annotation file (no sleep stages).")
    # mastersheet = mastersheet[~mastersheet['annot_path'].isna()]
    # # Exclude the 5 PSG wihtout ECG 
    # no_ECG = {
    #     ("S0001117385750", 1),
    #     ("S0001114925101", 1),
    #     ("S0001120767287", 2),
    #     ("S0001121582430", 3),
    #     ("S0001121903400", 1)
    # }
    # print(mastersheet[mastersheet.apply(lambda row: (row["sub_id"], row["session"]) in no_ECG, axis=1)]," PSG without ECG signal.")
    # mastersheet = mastersheet[~mastersheet.apply(lambda row: (row["sub_id"], row["session"]) in no_ECG, axis=1)]
    # print(len(mastersheet[mastersheet.apply(lambda row: (row["sub_id"], row["session"]) in too_early_start, axis=1)])," PSG with sleep stages starting too_early.")
    # mastersheet = mastersheet[mastersheet.apply(lambda row: (row["sub_id"], row["session"]) in too_late_start, axis=1)]
    # mastersheet = mastersheet[:1]
    rows = mastersheet.to_dict(orient="records")

    # --- Step 3: Convert EDF to H5 file --- 
    if config.output.overwrite_h5:
        # Convert all rows
        edf_to_h5(config, rows)
    else:
        # Only convert missing H5 files
        rows_to_process = [row for row in rows if not Path(row["h5_path"]).
                           exists()]
        print(len(rows_to_process), "subjects missing h5:", [row["sub_id"] for row in rows_to_process])
        if rows_to_process:
            edf_to_h5(config, rows_to_process)
        else:
            print("[INFO] Skipping EDF → h5: All files already exist and overwrite_h5 not allowed.")
        
    # --- Step 4: Extract selected features from the H5 file ---
    results = process_all_subjects(config, rows)

    # Collect failed subjects/sessions
    failed = [
        (row["sub_id"], row.get("session", None))
        for row, res in zip(rows, results)
        if res is None
    ]
    failed_df = pd.DataFrame(failed, columns=["sub_id", "session"])
    if not config.dataset.session is None:
        failed_csv_name = f"{config.dataset.name}_ses_{config.dataset.session}_failed_subjects.csv"
    else:
        failed_csv_name = f"{config.dataset.name}_failed_subjects_final.csv"
    failed_df.to_csv(failed_csv_name, index=False)
    print(f"[INFO] Saved failed subjects list to failed_subjects.csv")
    print(f"[INFO] {len(failed)} subjects failed:")
    for sub_id, session in failed:
        print(f"    - {sub_id}, session={session}")


def process_all_subjects(config, rows):
    results = []
    
    if config.run.parallel:
        workers = config.run.num_workers 
        print(f"[INFO] Processing {len(rows)} subjects in parallel with {workers} workers")
        with ProcessPoolExecutor(max_workers=workers) as executor:
            func = partial(_process_subject_safe, config)
            results = list(executor.map(func, rows))
    else:
        print(f"[INFO] Processing {len(rows)} subjects (not in parallel).")
        for row in rows:
            results.append(_process_subject_safe(config, row))
    
    return results


def _process_subject_safe(config, row):
    """Wrapper with error handling for one subject."""
    sub_id = row["sub_id"]
    session = row["session"]
    psg_id = f"sub-{sub_id}_ses-{session}" 
    try:
        extracted_features = process_subject(config, row)
        if extracted_features is None:
            print(f"[WARNING] No features extracted for subject {psg_id}", flush=True)
            return None
    except Exception as e:
        print(f"[ERROR] Failed processing for subject {psg_id}: {e}", flush=True)
        traceback.print_exc(file=sys.stdout) 
        sys.stdout.flush()
        return None
    else:
        print(f"[SUCCESS] Processing completed for subject {psg_id}")
        return extracted_features


# --- Step 4: Extract selected features from the H5 file ---
def process_subject(config, row):
    sub_id = row["sub_id"]
    session = row["session"]
    psg_id = f"sub-{sub_id}_ses-{session}" 

    # --- Step 4a: Define subject specific data ---
    if config.run.verbose:
        print(f"\nProcessing subject {row['sub_id']} session {row['session']}")
    project_root = Path(__file__).resolve().parents[1] 
    tmp_dir = project_root / "tmp" 
    tmp_dir_sub = tmp_dir / psg_id 
    tmp_dir_sub.mkdir(parents=True, exist_ok=True)

    # --- Step 4b: Read annotation XML file ---
    full_sleep_stages, df_events = read_annot(
        row,
        dataset_name=config.dataset.name)
    
    # Check if full_sleep_stages does NOT contain any 1, 2, 3, or 4
    if not any(s in [1, 2, 3, 4] for s in full_sleep_stages if not math.isnan(s)):
        print(f"[ERROR] {psg_id }: full_sleep_stages contains no valid stages!", flush=True)
        return None
    
    # --- Step 4c: Save events DataFrame ---
    if config.output.overwrite_events:
        sub_key = f"{config.dataset.name.lower()}_ses-{row['session']}_sub-{row['sub_id']}"
        df_events.to_csv(Path(config.paths.events) / f"{sub_key}_events.csv",
                     index = False)

    # --- Step 4d: Get sleep stages only for the night part ---
    sleep_stages, sleep_onset_time = process_sleep_stages(
        full_sleep_stages, 
        sfreq_global = row["sfreq_global"],
        start_time = row["start_time"],
        verbose = config.run.verbose)

    # --- Step 4e: Process Signals --- 
    processed_signals = process_signals(
        config, 
        row,
        full_sleep_stages, 
        tmp_dir_sub)

    # --- Step 4f: Only select windows for features that require them ---
    windows_dict = None
    if config.features.extract_all or any(f in config.features.selected for f in ["hrv", "cpc"]):
        windows_dict = selecting_windows(
            config,
            row,
            sleep_stages, 
            sleep_onset_time,
            processed_signals)

    # --- Step 4g: Exrtracted selected features --- 
    extracted_features = extract_features(config, row, tmp_dir_sub, sleep_stages, sleep_onset_time, processed_signals, df_events, windows_dict)

    # --- Step 4h: Save results ---
    save_features(config, row, extracted_features)
    save_features_wide(config, row, extracted_features)

    # --- Step 4i : Clean subject temporary folder ---
    if tmp_dir_sub.exists():
        shutil.rmtree(tmp_dir_sub) 

    return extracted_features

if __name__ == "__main__":
    main()
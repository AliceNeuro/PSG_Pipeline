import math
import xmltodict
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, time
from itertools import groupby

def read_annot(row, dataset_name):
    annot_path = row["annot_path"]
    if not pd.isna(annot_path) and Path(annot_path).suffix.lower() == ".xml":
        return read_annot_XML(row, dataset_name)
    elif dataset_name == "hsp_bidmc":
        return read_annot_BIDMC(row)
    elif dataset_name == "hsp_mgb":
        return read_annot_MGB(row)
    else:
        print("No read annot definition found.")
        return None, None


def read_annot_XML(row, dataset_name):
    sub_id = row["sub_id"]
    session = row["session"]
    psg_id = f"sub-{sub_id}_ses-{session}" 
    annot_path = row["annot_path"]
    sfreq_global = row["sfreq_global"]
    duration_samples = row["duration_samples"]

    # -------- EVENT ---------
    # Open and read xml 
    with open(annot_path, encoding='utf-8') as f:
        info_dict = xmltodict.parse(f.read())

    events = info_dict['CMPStudyConfig']['ScoredEvents']
    if events is None:
        print(f"[WARNING] {psg_id}: No events (other than sleep stages) in annotation file.")
        df_events = pd.DataFrame(data={
            'onset': [], 'duration': [], 'event_type': [],
            'channel': [], 'lowest_spo2': [], 'desaturation': []
        })
    else:
        events = events['ScoredEvent']
        if not isinstance(events, list):
            events = [events]
        df_events = pd.DataFrame(events)

        # Convert column names to lowercase for consistent processing
        df_events.columns = [col.lower() for col in df_events.columns]

        # Rename columns to match BIDS-style event.csv
        df_events = df_events.rename(columns={
            'start': 'onset',
            'name': 'event_type',
            'input': 'channel',
            'lowestspo2': 'lowest_spo2'
        })

        # Ensure correct types
        df_events['onset'] = df_events['onset'].astype(float)
        df_events['duration'] = df_events['duration'].astype(float)
        df_events['event_type'] = df_events['event_type'].astype(str)
        df_events['event_type'] = df_events['event_type'].str.lower().str.replace(' ', '_')
        if 'channel' in df_events.columns:
            df_events['channel'] = df_events['channel'].astype(str)
            df_events['channel'] = df_events['channel'].str.lower().str.replace(' ', '_') 

        # Specific to MrOS
        if dataset_name.lower() == "mros":
            df_events.loc[df_events.event_type=='hypopnea','event_type'] = 'hypopnea_(airflow_reduction30-50%)'
            df_events.loc[df_events.event_type=='unsure','event_type'] = 'hypopnea_(airflow_reduction>50%)'

        # Sort and reorder columns
        df_events = df_events.sort_values('onset', ignore_index=True, ascending=True)
        desired_order = ['onset', 'duration', 'event_type', 'channel', 'lowest_spo2', 'desaturation']
        existing_cols = [c for c in desired_order if c in df_events.columns]
        df_events = df_events[existing_cols]
        
    # -------- SLEEP STAGE ----------
    # Read and map full sleep stage (no only night) to match AASM guidelines  
    raw_stages = info_dict['CMPStudyConfig']['SleepStages']['SleepStage']
    stage_dict = {'0':0, '1':1, '2':2, '3':3, '4':3, '5':4}  
    sleep_stages_mapped = [stage_dict.get(str(s), np.nan) for s in raw_stages]
    
    # Create a df_stages
    epoch_length_sec = float(info_dict['CMPStudyConfig']['EpochLength'])
    n_epochs = int(duration_samples / (epoch_length_sec * sfreq_global))
    full_sleep_stages = np.array([s for s, g in groupby(sleep_stages_mapped) for _ in range(int(len(list(g)) * epoch_length_sec * sfreq_global))])

    # Warnings and adjustments for length mismatches
    if len(full_sleep_stages) != duration_samples:
        if len(full_sleep_stages) > duration_samples:
            # Too long → trim the extra part
            to_trim = full_sleep_stages[duration_samples:]
            if not np.all(np.isnan(to_trim)):
                print(f"[WARNING] {psg_id}: Trimming extra sleep_stages: {to_trim}")
            full_sleep_stages = full_sleep_stages[:duration_samples]
        else:
            # Too short → pad with NaNs if within one epoch
            missing_len = duration_samples - len(full_sleep_stages)
            if missing_len < (epoch_length_sec * sfreq_global): 
                # uncomplete epoch has no sleep stage so just np.nan
                full_sleep_stages = np.concatenate([full_sleep_stages, np.full(missing_len, np.nan)])
            else:
                print (f"[ERROR] {psg_id}: Sleep_stages is too short ({len(full_sleep_stages)}) for {duration_samples} total samples.")
    
    # Create DataFrame for sleep stages
    df_stages = pd.DataFrame({
        'onset': np.arange(0, n_epochs * 30, 30),
        'duration': np.full(n_epochs, 30),
        'sleep_stage': sleep_stages_mapped 
    })

    # -------- CONCAT SLEEP EVENTS ----------
    df_stages = df_stages.assign(event_type=np.nan, channel=np.nan, lowest_spo2=np.nan, desaturation=np.nan)
    df_events = df_events.assign(sleep_stage=np.nan)
    df_events = pd.concat([df_stages, df_events], ignore_index=True).sort_values('onset').reset_index(drop=True)

    return full_sleep_stages, df_events


def read_annot_BIDMC(row):
    sub_id = row["sub_id"]
    session = row["session"]
    psg_id = f"sub-{sub_id}_ses-{session}" 
    annot_path = row["annot_path"]
    sleep_stage_path = row["sleep_stage_path"]
    sfreq_global = row["sfreq_global"]
    duration_samples = row["duration_samples"]

    # -------- EVENT ---------
    if pd.isna(annot_path): # Creating empty df_events
        print(f"[WARNING] {psg_id}: No Annot Stage Path found.") # 1 time
        df_events = pd.DataFrame(columns=["onset", "duration", "event_type"])
    else: # Read the slepe stage from the specific file 
        df_annot = pd.read_csv(annot_path)

        required_cols = {"Epoch", "Record Time", "Time", "Length", "Description"}
        if required_cols.issubset(df_annot.columns):
            df_events = df_annot[list(required_cols)].copy()
            df_events = df_events.rename(columns={
                "Epoch": "epoch",
                "Record Time": "onset",
                "Time" : "clock_time",
                "Length": "duration",
                "Description": "event_type"
                })
            
            # Convert 'duration' → numeric (replace '-' or missing with 0.0)
            df_events['duration'] = pd.to_numeric(df_events['duration'], errors='coerce').fillna(0.0)

            # Convert 'onset' (HH:MM:SS or timedelta-like) → seconds
            df_events['onset'] = df_events['onset'].apply(datetime_to_sec)

            # Compute drops
            times = df_events['onset'].values
            drops = [i for i in range(1, len(times)) if times[i] < times[i - 1]]
            if len(drops) > 1:
                for drop_idx in drops:
                    drop_amount = times[drop_idx - 1] - times[drop_idx]
                    if drop_amount < 3600: # less than an hour drop 
                        row_to_move = df_events.iloc[drop_idx].copy()
                        df_events = df_events.drop(df_events.index[drop_idx]).reset_index(drop=True)
                        subset = df_events.iloc[:drop_idx].copy()
                        insert_idx = subset['onset'].searchsorted(row_to_move['onset'])

                        # Split df_events and insert the row
                        df_events = pd.concat([
                            df_events.iloc[:insert_idx],
                            pd.DataFrame([row_to_move]),
                            df_events.iloc[insert_idx:]
                        ]).reset_index(drop=True)

                        # print(f"[INFO] {psg_id}: Move event '{row_to_move['event_type']}' at index {drop_idx} to new index {insert_idx}")
            
            # Only one drop → standard post-midnight adjustment
            df_events['onset'] = ensure_post_midnight(df_events['onset'])

            # Shift event file if not matching recording time
            edf_start_sec = datetime_to_sec(row["start_time"]) 
            if edf_start_sec < 5*3600:
                print(f"[WARNING] {psg_id}: EDF start after midnight: {edf_start_sec} seconds.")
                edf_start_sec += 24*3600
            df_events['clock_time'] = df_events['clock_time'].apply(datetime_to_sec)   

            # If first value already after midnight add 24h 
            first_idx_mask = df_events['clock_time'].notna() & (df_events['clock_time'] != 0)
            first_idx = first_idx_mask.idxmax() if first_idx_mask.any() else 0
            df_events['adjusted_clock_time'] = df_events['clock_time']
            if df_events.loc[first_idx, 'adjusted_clock_time'] < 5*3600:
                print(f"[WARNING] {psg_id}: EVENT start after midnight: {df_events.loc[first_idx, 'adjusted_clock_time']} seconds.")
                df_events.loc[first_idx, 'adjusted_clock_time'] += 24*3600
            df_events['adjusted_clock_time'] = ensure_post_midnight(df_events['adjusted_clock_time'])

            # Find first index
            first_after_start_idx = df_events.index[df_events['adjusted_clock_time'] >= edf_start_sec][0]
            subset = df_events.iloc[: first_after_start_idx + 5].copy()
            offset = subset['adjusted_clock_time'].astype(float) - subset['onset'].astype(float)
            subset['time_offset'] = np.where(offset != 0, offset, np.nan)
            offset_diffs = subset['time_offset'].diff().abs()
            offset_change_idxs = offset_diffs[offset_diffs.gt(1)].index
            n_offsets = len(offset_change_idxs)
            change_first_epoch = False
            first_correct_epoch = 1

            if n_offsets >= 1: # weird time at first
                last_change_idx = offset_change_idxs[-1]   # <-- only use the last one
                print(f"[WARNING] {psg_id}: Detected {n_offsets} time offset jumps. Using the last one at index {last_change_idx}.")
                correct_start = subset.loc[last_change_idx, 'time_offset']
                # rows BEFORE the last change need correction
                to_fix = df_events.iloc[:last_change_idx].copy()
                to_fix['onset'] = (to_fix['adjusted_clock_time'] - correct_start)
                if (to_fix['onset'] > (24*3600)).any():
                    to_fix['onset'] = to_fix['onset'] - (24*3600)

                # rows AFTER the last change are already correct
                df_events = pd.concat([to_fix, df_events.iloc[last_change_idx:]]).reset_index(drop=True)
                first_correct_epoch = subset.loc[last_change_idx, 'epoch']
                change_first_epoch = True
                print(f"[INFO] {psg_id}: New first epoch: {first_correct_epoch}")

            annot_start_sec = df_events['adjusted_clock_time'].iloc[first_idx] - df_events['onset'].iloc[first_idx]
            if (
                (edf_start_sec is not None) and not np.isnan(edf_start_sec) and
                (annot_start_sec is not None) and not np.isnan(annot_start_sec)
                ):
                offset_sec = annot_start_sec - edf_start_sec 
                if abs(offset_sec) <= 1:
                    df_events["onset"] = df_events["onset"].astype(float) + offset_sec
                    if int(offset_sec) != 0:
                        print(f"[OK] {psg_id}: Shift of {offset_sec} s")
                elif offset_sec < -1:
                    print(f"[WARNING] {psg_id}: Event start before recording ({offset_sec:.3f}s)")
                    df_events["onset"] = df_events["onset"].astype(float) + offset_sec
                else:
                    print(f"[WARNING] {psg_id}: offset_event >1s ({offset_sec:.3f}s)")
                    df_events["onset"] = df_events["onset"].astype(float) + offset_sec
            else:
                print(f"[ERROR] {psg_id}: edf_start ({edf_start_sec}) or annot_start ({annot_start_sec}) is not defined")
            
            if change_first_epoch:
                first_pos_onset = df_events.loc[df_events["onset"] > 0, "onset"].iloc[0]
                epoch_shift = int(first_pos_onset // 30)
                first_correct_epoch -= epoch_shift
                if epoch_shift != 0:
                    print(f"[INFO] {psg_id}: Shifted first epoch: {first_correct_epoch}")
            
        else:
            print(f"[WARNING] {psg_id}:  No events extracted - Missing expected columns: {required_cols - set(df_annot.columns)}")
            df_events = pd.DataFrame(columns=["event_type", "onset", "duration"])
    
    # Final type enforcement
    df_events['onset'] = df_events['onset'].astype(float)
    df_events['duration'] = df_events['duration'].astype(float)
    df_events['event_type'] = (
        df_events['event_type']
        .astype(str)
        .str.lower()
    )

    # Sort and reorder columns
    df_events = df_events.drop(columns = ['clock_time'])
    df_events = df_events.sort_values('onset', ignore_index=True, ascending=True)
    desired_order = ['onset', 'duration', 'event_type']
    existing_cols = [c for c in desired_order if c in df_events.columns]
    df_events = df_events[existing_cols]
    
    # -------- SLEEP STAGE ----------
    # Open and read sleep CSV 
    if pd.isna(sleep_stage_path): # Should never be the case for BIDMC
        print(f"[ERROR] {psg_id}: No Sleep Stage Path found.")
        full_sleep_stages = []
    else:
        df_sleep = pd.read_csv(sleep_stage_path)
        start_idx = df_sleep.index[df_sleep["Epoch"] == first_correct_epoch][0]
        df_sleep = df_sleep.iloc[start_idx:].copy()
        stage_col = next((c for c in df_sleep.columns if "stage" in c.lower()), None)
        if stage_col is None:
            print(f"[ERROR] {psg_id}: No column containing 'stage' found in df_sleep")
            return None, None 
        raw_stages = df_sleep[stage_col]
        stage_dict = {"W": 0, "WAKE": 0, 
                      "N1": 1, "1": 1,
                      "N2": 2, "2": 2,
                      "N3": 3, "N4": 3, "3": 3, "4": 3,
                      "R": 4, "REM": 4}
        sleep_stages_mapped = [stage_dict.get(str(s), np.nan) for s in raw_stages]

        ### Adding a check 
        first_epoch = df_sleep.iloc[0]["Epoch"]
        if first_epoch != 1:
            print(f"[WARNING] {psg_id}: First Epoch in df_sleep is: {first_epoch}")

        # Create a df_stages
        epoch_length_sec = 30
        n_epochs = int(duration_samples / (epoch_length_sec * sfreq_global))
        full_sleep_stages = np.array([s for s, g in groupby(sleep_stages_mapped) for _ in range(int(len(list(g)) * epoch_length_sec * sfreq_global))])

        # Warnings and adjustments for length mismatches
        if len(full_sleep_stages) != duration_samples:
            if len(full_sleep_stages) > duration_samples:
                # Adjust full sleep stages AND sleep_stages_mapped
                excess_samples = len(full_sleep_stages) - duration_samples
                samples_per_epoch = int(epoch_length_sec * sfreq_global)
                excess_epochs = int(np.ceil(excess_samples / samples_per_epoch))
                if excess_epochs > 0:
                    to_trim = sleep_stages_mapped[-excess_epochs:]
                    if excess_epochs > 1 and any(not (np.isnan(x) or x == 0) for x in to_trim):
                        print(f"[WARNING] {psg_id}: Trimming {excess_epochs} extra sleep_stages: {to_trim}")
                    sleep_stages_mapped = sleep_stages_mapped[:-excess_epochs]
                full_sleep_stages = full_sleep_stages[:duration_samples]
            else:
                # Too short → pad with NaNs if within one epoch
                missing_len = duration_samples - len(full_sleep_stages)
                if missing_len >= (epoch_length_sec * sfreq_global):
                    missing_epochs = int(np.ceil(missing_len / (epoch_length_sec * sfreq_global)))
                    sleep_stages_mapped = np.concatenate([sleep_stages_mapped, np.full(missing_epochs, np.nan)])
                    print(f"[ERROR] {psg_id}: Sleep_stages too short "
                          f"({len(full_sleep_stages)} vs {duration_samples} samples) adding {missing_epochs} NaN stages.")

                # uncomplete epoch has no sleep stage so just np.nan
                full_sleep_stages = np.concatenate([full_sleep_stages, np.full(missing_len, np.nan)])

        # Create DataFrame for sleep stages
        if (len(sleep_stages_mapped) - n_epochs) == 1:
            sleep_stages_mapped = sleep_stages_mapped[:-1] # remove uncomplete 
        try:
            df_stages = pd.DataFrame({
                'onset': np.arange(0, n_epochs * 30, 30),
                'duration': np.full(n_epochs, 30),
                'sleep_stage': sleep_stages_mapped
            })
        except Exception as e:
            print(f"[ERROR] PSG {psg_id}: n_epoch and sleep_stage not same length ({e})")
            df_stages = pd.DataFrame()
        
    # -------- CONCAT SLEEP EVENTS ----------
    df_stages = df_stages.assign(event_type=np.nan)
    df_events = df_events.assign(sleep_stage=np.nan)
    df_events = pd.concat([df_stages, df_events], ignore_index=True).sort_values('onset').reset_index(drop=True)

    return full_sleep_stages, df_events



def read_annot_MGB(row):
    sub_id = row["sub_id"]
    session = row["session"]
    psg_id = f"sub-{sub_id}_ses-{session}" 
    annot_path = row["annot_path"]
    sleep_stage_path = row["sleep_stage_path"]
    sfreq_global = row["sfreq_global"]
    duration_samples = row["duration_samples"]

    # ------------------------
    # -------- EVENT ---------
    # ------------------------
    if pd.isna(annot_path): # Creating empty df_events
        print(f"[WARNING] {psg_id}: No annot path found.") # 1 time
        df_events = pd.DataFrame(columns=["onset", "duration", "event_type"])
    else: # Read the slepe stage from the specific file 
        df_annot = pd.read_csv(annot_path)
        required_cols_psg = {"epoch", "time", "duration", "event"}
        if required_cols_psg.issubset(df_annot.columns):
            df_events = df_annot[list(required_cols_psg)].copy()
            df_events = df_events.rename(columns={
                "time" : "clock_time",
                "event": "event_type"
                })
            
            # Convert 'duration' → numeric (replace '-' or missing with 0.0)
            df_events['duration'] = pd.to_numeric(df_events['duration'], errors='coerce').fillna(0.0)
         
            # Shift event file if not matching recording time
            edf_start_sec = datetime_to_sec(row["start_time"]) 
            # Nope in the end it's not impacting normally 
            # if "Recording Resumed" in df_events['event_type'].values:
            #     # Cut the table so this row is the first
            #     resumed_row = df_events[df_events['event_type'] == "Recording Resumed"].iloc[0]
            #     df_events = df_events.loc[resumed_row.name:].reset_index(drop=True)
            #     # Update edf_start_sec to the clock_time of this row
            #     edf_start_sec = datetime_to_sec(resumed_row['clock_time'])

            df_events['clock_time'] = df_events['clock_time'].apply(datetime_to_sec)

            # --- START: ENSURING TIME CONTINUITY - MIDNIGHT - LIGHT ---      
            # Find rows with "LIGHT" in 'event_type' 
            mask_light = df_events['event_type'].str.contains("LIGHT", case=True, na=False)
            light_rows = df_events[mask_light].copy()  # store for later
            df_events = df_events[~mask_light].reset_index(drop=True)  # remove from df_events

            # Get first idx here without LIGHT event added
            first_idx_mask = df_events['clock_time'].notna() & (df_events['clock_time'] != 0)
            first_idx = first_idx_mask.idxmax() if first_idx_mask.any() else 0
            annot_start_sec = df_events['clock_time'].iloc[first_idx]

            # Get first sleep idx (in case event before sleep stages)
            first_idx_sleep_mask = (
                df_events['clock_time'].notna() &
                (df_events['clock_time'] != 0) &
                df_events['event_type'].str.lower().str.contains(r'^sleep|^stage', regex=True)
            )
            first_idx_sleep = first_idx_sleep_mask.idxmax() if first_idx_sleep_mask.any() else 0
            sleep_start_sec = df_events['clock_time'].iloc[first_idx_sleep]

            # Compute drops
            times = df_events['clock_time'].values
            drops = [i for i in range(1, len(times)) if times[i] < times[i - 1]]

            if len(drops) > 1:
                print(f"[INFO] {psg_id}: More than one drop even after removing LIGHT events.")
            else:
                # Only one drop → standard post-midnight adjustment
                df_events['clock_time'] = ensure_post_midnight(df_events['clock_time'])
            
            # Replace correclty the LIGHT events 
            for idx, row in light_rows.iterrows():
                ct = row['clock_time']
                
                # Add 24h if the light event occurs before noon
                if ct < 12*3600:
                    ct += 24*3600
                row['clock_time'] = ct

                # Find correct position to insert while keeping ascending order
                insert_idx_candidates = df_events.index[df_events['clock_time'] >= ct].tolist()

                if insert_idx_candidates:
                    insert_idx = insert_idx_candidates[0]  # first row >= ct
                else:
                    insert_idx = len(df_events)  # append at end if all smaller

                # If the correct position is 0, it safely handles first row
                df_events = pd.concat([
                    df_events.iloc[:insert_idx],
                    pd.DataFrame([row]),
                    df_events.iloc[insert_idx:]
                ]).reset_index(drop=True)
                
                # if abs(idx - insert_idx) > 1:   
                #     print(f"[INFO] {psg_id}: Move event '{row['event_type']}' at index {idx} to new index {insert_idx+1}")
            # --- END: ENSURING TIME CONTINUITY - MIDNIGHT - LIGHT ---    

            # --- START: SHIFT TO MATCH RECORDING START ---
            if (
                (edf_start_sec is not None) and not np.isnan(edf_start_sec) and
                (annot_start_sec is not None) and not np.isnan(annot_start_sec)
                ):
                offset_sec = annot_start_sec - edf_start_sec 
                if abs(offset_sec) <= 1:
                    if offset_sec != 0:
                        print(f"[INFO] {psg_id}: Based on first event - shift of {offset_sec} s")
                    df_events["onset"] = df_events["clock_time"].astype(float) - (edf_start_sec + offset_sec)
                elif offset_sec < -1:
                    # Check first sleep stage if event start before recording
                    new_offset_sec = sleep_start_sec - edf_start_sec
                    if abs(new_offset_sec) <= 1:
                        if new_offset_sec != 0:
                            print(f"[INFO] {psg_id}: Based on first stage - shift of {new_offset_sec} s")
                        df_events["onset"] = df_events["clock_time"].astype(float) - (edf_start_sec + new_offset_sec)
                    elif new_offset_sec < -1:
                        print(f"[WARNING] {psg_id}: Events ({offset_sec:.3f}s) and Sleep Stages ({new_offset_sec:.3f}s) start before recording.")
                        df_events["onset"] = df_events["clock_time"].astype(float) - (edf_start_sec)
                    else:
                        print(f"[ERROR] {psg_id}: Events start before recording ({offset_sec:.3f}s) and Sleep Stages too much after ({new_offset_sec:.3f}s).")
                        df_events = pd.DataFrame(columns=["onset", "duration", "event_type"]) 
                else:
                    print(f"[WARNING] {psg_id}: offset_event >1s ({offset_sec:.3f}s)")
                    df_events["onset"] = df_events["clock_time"].astype(float) - (edf_start_sec)
            else:
                print(f"[ERROR] {psg_id}: edf_start ({edf_start_sec}) or annot_start ({annot_start_sec}) is not defined")
            # --- END: SHIFT TO MATCH RECORDING START ---

            # --- START: DEVIDE IN DF_EVENTS AND DF_SLEEP ---
            df_events["event_flat"] = df_events["event_type"].str.lower().str.replace(r"[-_ ]", "", regex=True)
            mask_sleep = (
                df_events["event_flat"].str.startswith("sleepstage") |
                df_events["event_flat"].str.startswith("stage")
            )
            df_sleep = df_events[mask_sleep].copy()
            df_events = df_events[~mask_sleep].copy()
            # Final type enforcement for df_events
            df_events['onset'] = df_events['onset'].astype(float)
            df_events['duration'] = df_events['duration'].astype(float)
            df_events['event_type'] = (
                df_events['event_type']
                .astype(str)
                .str.lower()
            )

            # Sort and reorder columns
            df_events = df_events.drop(columns = ['epoch','clock_time', 'event_flat'])
            df_events = df_events.sort_values('onset', ignore_index=True, ascending=True)
            # --- END: DEVIDE IN DF_EVENTS AND DF_SLEEP ---

            # -------------------------------
            # -------- SLEEP STAGE ----------
            # -------------------------------
            # Check first sleep stage colck time - muliple of 30 or not
            if df_sleep.empty:
                print(f"[ERROR] {psg_id} No Sleep_Stages in event !!")
                df_sleep = pd.DataFrame(columns=["onset", "duration", "sleep_stage"]) 
                full_sleep_stages = np.full(duration_samples, np.nan) 
            else:
                # Get the map sleep stages 
                stage_dict = {"W": 0, "WAKE": 0, 
                      "N1": 1, "1": 1,
                      "N2": 2, "2": 2,
                      "N3": 3, "N4": 3, "3": 3, "4": 3,
                      "R": 4, "REM": 4}
                df_sleep['sleep_stage'] = (
                    df_sleep["event_flat"]
                    .str.replace(r"^(sleepstage|stage)", "", case=False, regex=True)
                    .str.upper()
                    .map(stage_dict)
                )
                
                # Repeat sleep stage every 30s epoch if only changes are annotated
                epoch_length_sec = 30
                if not (df_sleep["duration"] == epoch_length_sec).all():
                    print(f"[INFO] {psg_id}: Adjusting to get 30s epochs.")
                    # Append one extra endpoint for the final stage
                    all_onsets = np.append(df_sleep["onset"].values,
                                        df_sleep["onset"].iloc[-1] + epoch_length_sec)
                    # Repeat stage labels
                    epoch_counts = (np.diff(all_onsets) // epoch_length_sec).astype(int)
                    expanded_stages = np.repeat(df_sleep["sleep_stage"].values, epoch_counts)
                    # Construct the new DataFrame
                    expanded_onsets = (
                        np.arange(len(expanded_stages)) * epoch_length_sec  + df_sleep["onset"].iloc[0]
                    )
                    df_sleep = pd.DataFrame({
                        "onset": expanded_onsets,
                        "duration": epoch_length_sec,
                        "sleep_stage": expanded_stages
                    }).reset_index(drop=True)

                # Drop unecessary columns
                cols_to_drop = ['epoch', 'clock_time', 'event_type', 'event_flat']
                existing_cols = [c for c in cols_to_drop if c in df_sleep.columns]
                df_sleep = df_sleep.drop(columns=existing_cols)

                # If mulitple of 30 second than add all the missing sleep stages
                first_sleep_onset = df_sleep.iloc[0]['onset']
                if first_sleep_onset > 0: # add missing stages 
                    if abs(first_sleep_onset) % epoch_length_sec == 0: 
                        n_missing = int(first_sleep_onset // epoch_length_sec)
                        print(f"[INFO] {psg_id}: Adding {n_missing} nan epoch at the begining because first_sleep_onset = {first_sleep_onset}")
                        missing_df = pd.DataFrame({
                            "onset": np.arange(n_missing) * epoch_length_sec,
                            "duration": epoch_length_sec,
                            "sleep_stage": [np.nan] * n_missing
                        })
                        df_sleep = pd.concat([missing_df, df_sleep]).reset_index(drop=True)
                    else: 
                        print(f"[INFO] {psg_id}: First sleep stage onset not multiple of 30: {first_sleep_onset}")
                
                # If extra sleep at the begging remove them 
                if first_sleep_onset < 0: # remove extra stages 
                    n_extra = math.ceil(abs(first_sleep_onset) / epoch_length_sec)
                    extra = df_sleep["sleep_stage"].iloc[:n_extra]
                    print(f"[INFO] {psg_id}: {n_extra} sleep stages for first_sleep_onset={first_sleep_onset}")
                    if all((pd.isna(x) or x == 0) for x in extra):
                        df_sleep = df_sleep.iloc[n_extra:]
                        print(f"[INFO] {psg_id}: Removing first Sleep Stages: {extra.to_list()}")    
                    else: 
                        print(f"[ERROR] {psg_id}: Cannot remove first Sleep Stages that contain info: {extra.to_list()}")      
                        df_sleep = pd.DataFrame(columns=["onset", "duration", "sleep_stage"])      

                # Deduce full_sleep_stages
                full_sleep_stages = np.full(duration_samples, np.nan) # init
                overflow_values = []
                for i, row in df_sleep.iterrows():
                    start_idx = int(row['onset'] * sfreq_global)
                    end_idx = int((row['onset'] + epoch_length_sec) * sfreq_global) 
   
                    if start_idx >= duration_samples:
                        overflow_values.extend(df_sleep.loc[i:, 'sleep_stage'].values)
                        break  
                    
                    if end_idx > (duration_samples):
                        next_epoch = duration_samples + (epoch_length_sec *  sfreq_global)
                        if end_idx > next_epoch:
                            overflow_values.append(row['sleep_stage'])
                        end_idx = duration_samples
                    
                    full_sleep_stages[start_idx:end_idx] = row['sleep_stage']
                
                if overflow_values and any(not (np.isnan(x) or x == 0) for x in overflow_values):
                    overflow_values = [int(x) if not (isinstance(x, float) and math.isnan(x)) else np.nan for x in overflow_values]
                    print(f"[WARNING] {psg_id}: Trimming extra sleep_stages: {overflow_values}")

            # -------- CONCAT SLEEP EVENTS ----------
            df_sleep = df_sleep.assign(event_type=np.nan)
            df_events = df_events.assign(sleep_stage=np.nan)
            df_events = pd.concat([df_sleep, df_events], ignore_index=True).sort_values('onset').reset_index(drop=True)
            desired_order = ['onset', 'duration', 'sleep_stage', 'event_type']
            existing_cols = [c for c in desired_order if c in df_events.columns]
            df_events = df_events[existing_cols]
        
        else:
            df_events = pd.DataFrame(columns=["onset", "duration", "event_type"])
            full_sleep_stages = np.full(duration_samples, np.nan) 
            print(f"[OTHER] {psg_id}: NEED XLTEK OR PSG ANNOT + 3 CORRECT COLS", flush=True)
    
    return full_sleep_stages, df_events




############ OTHERS UTIL FUNCTIONS ############ 

def datetime_to_sec(t_input):
    """
    Convert time-like input to seconds since midnight (float, including milliseconds).
    """

    if t_input is None or (isinstance(t_input, float) and np.isnan(t_input)):
        return np.nan

    # --- Convert to string to check for negative seconds ---
    t_str = str(t_input).strip()
    if '-' in t_str.split(' ')[-1]:  # negative seconds detected
        return 0.0

    # --- Handle datetime or time objects ---
    if isinstance(t_input, datetime):
        t_obj = t_input.time()
    elif isinstance(t_input, time):
        t_obj = t_input
    else:
        # Remove timezone suffix if present (e.g., +00:00, Z)
        t_str_clean = t_str.split('+')[0].split('Z')[0]

        t_obj = None
        # Try multiple formats
        for fmt in [
            "%Y-%m-%d %H:%M:%S.%f",
            "%Y-%m-%d %H:%M:%S",
            "%I:%M:%S %p",
            "%H:%M:%S.%f",
            "%H:%M:%S"
        ]:
            try:
                t_obj = datetime.strptime(t_str_clean, fmt).time()
                break
            except ValueError:
                continue

        if t_obj is None:
            print(f"[WARNING] Could not parse time: {t_input}")
            return np.nan

    # --- Compute seconds including fractional part ---
    t_sec = (
        t_obj.hour * 3600 +
        t_obj.minute * 60 +
        t_obj.second +
        t_obj.microsecond / 1e6
    )

    return t_sec


def ensure_post_midnight(times_sec: pd.Series) -> pd.Series:
    continuous_sec = []
    prev_sec = None
    rollover = 0  # track cumulative 24h rollovers

    for sec in times_sec:
        if pd.isna(sec):
            continuous_sec.append(np.nan)
            continue
        if prev_sec is not None and sec + rollover < prev_sec:
            # crossed midnight
            rollover += 24 * 3600
        continuous_sec.append(sec + rollover)
        prev_sec = sec + rollover

    return pd.Series(continuous_sec, index=times_sec.index)
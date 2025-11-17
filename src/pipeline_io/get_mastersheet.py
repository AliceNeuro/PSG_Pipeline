from pathlib import Path
import mne
import pandas as pd
from datetime import datetime, timezone
from itertools import islice
import warnings

########## MASTERSHEET FUNCTIONS ########## 
def get_mastersheet(config):

    # Mapping dataset name to its corresponding writer function
    MASTERSHEET_WRITERS = {
        "mros_ses-1": write_mastersheet_mros_ses1,
        "mros_ses-2": write_mastersheet_mros_ses2,
        "hsp_bidmc": write_mastersheet_hsp_bidmc,
        "hsp_mgb": write_mastersheet_hsp_mgb,
        "shhs_ses-1": write_mastersheet_shhs,
        "shhs_ses-2": write_mastersheet_shhs,
        "mesa": write_mastersheet_mesa,
    }

    # Build the mastersheet file name with correct f-string
    mastersheet_folder = Path(config.paths.mastersheets)
    dataset_key = (
        f"{config.dataset.name.lower()}_ses-{config.dataset.session}"
        if config.dataset.session is not None
        else config.dataset.name.lower()
    )
    mastersheet_paths = [
        mastersheet_folder / f"{dataset_key}_mastersheet_post_summary.csv",
        mastersheet_folder / f"{dataset_key}_mastersheet.csv"
    ]

    # Load the first existing mastersheet if overwrite is False
    for path in mastersheet_paths:
        if path.exists() and not config.output.overwrite_mastersheet:
            print(f"[INFO] Loading existing mastersheet from {path}")
            return read_mastersheet(path)

    # Error if no mastersheet definition 
    if dataset_key not in MASTERSHEET_WRITERS:
        raise NameError(f"[ERROR] No mastersheet writer defined for dataset/session '{dataset_key}'")

    # Create and save the mastersheet 
    writer = MASTERSHEET_WRITERS[dataset_key]
    mastersheet_path = mastersheet_folder / f"{dataset_key}_mastersheet.csv"
    mastersheet = writer(config, mastersheet_path)

    # Convert start_time to correct format and timezone
    if "start_time" in mastersheet.columns:
        mastersheet["start_time"] = [
            datetime.fromisoformat(str(x)).replace(tzinfo=timezone.utc)
            if pd.notna(x) else None
            for x in mastersheet["start_time"]
        ]
    
    mastersheet["start_time"] = mastersheet["start_time"].astype(str)

    return mastersheet

def read_mastersheet(mastersheet_path):
    mastersheet = pd.read_csv(mastersheet_path)
    return mastersheet

def write_mastersheet_sub(config, sub_id, site, edf_path, annot_path, session, sleep_stage_path = None):
    # Load raw file without preloading data
    if edf_path is None:
        start_time = None
        sfreq_global = None
        duration_samples = None
        duration_sec = None
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            raw = mne.io.read_raw_edf(str(edf_path), preload=False, verbose=False)
        start_time = raw.info['meas_date']
        sfreq_global = raw.info['sfreq']
        duration_samples = raw.n_times 
        duration_sec = raw.n_times / sfreq_global 

    # Add h5 and events path
    dataset_key = f"{config.dataset.name.lower()}_ses-{session}"

    h5_path = Path(config.paths.h5_data) / f"{dataset_key}_sub-{sub_id}_signals.h5"
    events_path = Path(config.paths.events) / f"{dataset_key}_sub-{sub_id}_events.csv"
    extracted_features_path = Path(config.paths.extracted_features) / f"{dataset_key}_sub-{sub_id}_extracted_features.csv"

    record = {
        "sub_id": sub_id,
        "dataset": config.dataset.name,
        "site": site,
        "session": session,
        "start_time": start_time,
        "sfreq_global": sfreq_global,
        "duration_samples": duration_samples,
        "duration_sec": duration_sec,
        "edf_path": str(edf_path) if edf_path is not None else None,
        "annot_path": str(annot_path) if annot_path is not None else None,
        "h5_path": str(h5_path) if h5_path is not None else None,
        "events_path": str(events_path) if events_path is not None else None,
        "extracted_features_path": str(extracted_features_path) if extracted_features_path is not None else None,
        "sleep_stage_path": str(sleep_stage_path) if sleep_stage_path is not None else None
    }
  
    return record
    
def write_mastersheet_mros_ses1(config, mastersheet_path):
    records = []
    session = int(config.dataset.session)
    for site_folder in Path(config.paths.input).iterdir():
        if not site_folder.is_dir():
            continue 

        for file in Path(site_folder).iterdir():
            if file.suffix.lower() != ".edf":
                    continue
            
            sub_id = file.stem
            site = sub_id[:2]
            edf_path = site_folder / f"{sub_id}.edf"
            annot_path = site_folder / f"{sub_id}.edf.XML"

            # Handle problematic annotation filename
            subset_xml_filenames = {
                "bi0694": "bi/bi0694.edu.XML",
                "bi0848": "bi/bi0848.edu.XML",
                "mn1786": "mn/mn1786.XML"}
            if sub_id in subset_xml_filenames:
                annot_path = Path(config.paths.input) / Path(subset_xml_filenames[sub_id])

            # Check files exist
            if edf_path is None:
                print(f"[WARNING] Missing EDF for sub-{sub_id}_ses{session}")
            if annot_path is None:
                print(f"[WARNING] Missing Annotations for sub-{sub_id}_ses{session}")
            
            try: 
                record = write_mastersheet_sub(config, sub_id, site, edf_path, annot_path, session)
            except Exception as e:
                print(f"[ERROR] Cannot read {edf_path.name}: {e}")
                continue   

            records.append(record)
        
    # Convert to DataFrame
    mastersheet = pd.DataFrame(records)
    mastersheet = mastersheet.sort_values("sub_id")
    mastersheet.to_csv(mastersheet_path, index=False)
    print(f"✅ Saved mastersheet to {mastersheet_path}")
    
    return mastersheet

def write_mastersheet_mros_ses2(config, mastersheet_path):
    records = []
    session = int(config.dataset.session)

    for sub_folder in Path(config.paths.input).iterdir():
        if not sub_folder.is_dir():
            continue 
        
        folder_name = sub_folder.name
        sub_id = folder_name.split("_")[0]
        site = sub_id[:2]

        # Get Paths
        all_files = list(sub_folder.iterdir())
        edf_path = next((f for f in all_files if f.suffix.lower() == ".edf"), None)
        annot_path = next((f for f in all_files if f.name.lower().endswith(".edf.xml")), None)

        # Check if both paths are found
        if edf_path is None:
            print(f"[WARNING] Missing EDF for sub-{sub_id}_ses{session}")
            continue
        if annot_path is None:
            print(f"[WARNING] Missing Annotations for sub-{sub_id}_ses{session}")
        
        try: 
            record = write_mastersheet_sub(config, sub_id, site, edf_path, annot_path, session)
        except Exception as e:
            print(f"[ERROR] Cannot read {edf_path.name}: {e}")
            continue  
          
        records.append(record)

    # Convert to DataFrame
    mastersheet = pd.DataFrame(records)
    mastersheet = mastersheet.sort_values("sub_id")
    mastersheet.to_csv(mastersheet_path, index=False)
    print(f"✅ Saved mastersheet to {mastersheet_path}")
    
    return mastersheet


def write_mastersheet_hsp_bidmc(config, mastersheet_path):
    records = []

    #for sub_folder in islice(Path(config.paths.input).iterdir(), 100):
    for sub_folder in Path(config.paths.input).iterdir():
        if not sub_folder.is_dir():
            continue 

        folder_name = sub_folder.name
        sub_id = folder_name.split("-")[1]
        site = "bidmc"

        all_sessions = [f for f in sub_folder.iterdir() if f.is_dir()]
        for ses in all_sessions:
            ses_folder = sub_folder / ses / "eeg"
            session = int(str(ses).split("-")[-1])

            # Get Paths
            all_files = list(ses_folder.iterdir())
            edf_path = next((f for f in all_files if f.suffix.lower() == ".edf"), None)
            annot_path = next((f for f in all_files if f.name.lower().endswith("_events_annotations.csv")), None)
            sleep_stage_path = next((f for f in all_files if f.name.lower().endswith("_sleep_annotations.csv")), None)

            # Check if both paths are found
            if edf_path is None:
                print(f"[WARNING] Missing EDF for sub-{sub_id}_ses{session}")
                continue
            if annot_path is None:
                print(f"[WARNING] Missing Annotations for sub-{sub_id}_ses{session}")
            
            try: 
                record = write_mastersheet_sub(config, sub_id, site, edf_path, annot_path, session, sleep_stage_path)
            except Exception as e:
                print(f"[ERROR] Cannot read {edf_path.name}: {e}")
                continue  
          
            records.append(record)

    # Convert to DataFrame
    mastersheet = pd.DataFrame(records)
    mastersheet = mastersheet.sort_values("sub_id")
    mastersheet.to_csv(mastersheet_path, index=False)
    print(f"✅ Saved mastersheet to {mastersheet_path}")
    
    return mastersheet



def write_mastersheet_hsp_mgb(config, mastersheet_path):
    records = []

    #for sub_folder in islice(Path(config.paths.input).iterdir(), 10):
    for sub_folder in Path(config.paths.input).iterdir():
        if not sub_folder.is_dir():
            continue 

        folder_name = sub_folder.name
        sub_id = folder_name.split("-")[1]
        site = "mgb"

        all_sessions = [f for f in sub_folder.iterdir() if f.is_dir()]
        for ses in all_sessions:
            ses_folder = sub_folder / ses / "eeg"
            session = int(str(ses).split("-")[-1])

            # Get Paths
            all_files = list(ses_folder.iterdir())
            edf_path = next((f for f in all_files if f.suffix.lower() == ".edf"), None)

            # potential issues for annotation file
            csv_files = [f for f in all_files if f.name.lower().endswith(".csv")]
            if not csv_files:
                annot_path = None
            else:
                annot_path = next((f for f in csv_files if f.name.lower().endswith("annotations.csv")), None)
                if annot_path is None:
                    annot_path = next((f for f in csv_files if f.name.lower().endswith("xltek.csv")), None)
                if annot_path is None:
                    annot_path = csv_files[0]

            # Check if both paths are found
            if edf_path is None:
                print(f"[WARNING] Missing EDF for sub-{sub_id}_ses{session}")
                continue
            if annot_path is None:
                print(f"[WARNING] Missing Annotations for sub-{sub_id}_ses{session}")
            
            try: 
                record = write_mastersheet_sub(config, sub_id, site, edf_path, annot_path, session)
            except Exception as e:
                print(f"[ERROR] Cannot read {edf_path.name}: {e}")
                continue  
          
            records.append(record)

    # Convert to DataFrame
    mastersheet = pd.DataFrame(records)
    mastersheet = mastersheet.sort_values("sub_id")
    mastersheet.to_csv(mastersheet_path, index=False)
    print(f"✅ Saved mastersheet to {mastersheet_path}")
    
    return mastersheet


def write_mastersheet_shhs(config, mastersheet_path):
    records = []
    session = int(config.dataset.session)
    site =f"shhs{str(session)}" 
    edf_folder = Path(config.paths.input) / "edfs" / f"shhs{str(session)}" 
    annot_folder = Path(config.paths.input) / "annotations-events-profusion" / f"shhs{str(session)}" 

    if not edf_folder.is_dir():
        print("[ERROR] No edfs folder.")
    if not annot_folder.is_dir():
        print("[ERROR] No annotation profusion nsrr folder.")
    
    for edf_path in Path(edf_folder).iterdir():
        sub_id = edf_path.stem
        annot_file = f"{sub_id}-profusion.xml"
        annot_path = Path(annot_folder) / annot_file
        
        # Check files exist
        if edf_path is None:
            print(f"[WARNING] Missing EDF for sub-{sub_id}_ses{session}")
            continue
        if annot_path is None:
            print(f"[WARNING] Missing Annotations for sub-{sub_id}_ses{session}")

        try: 
            record = write_mastersheet_sub(config, sub_id, site, edf_path, annot_path, session)
            
        except Exception as e:
            print(f"[ERROR] Cannot read {edf_path.name}: {e}")
            continue   

        records.append(record)
        
    # Convert to DataFrame
    mastersheet = pd.DataFrame(records)
    mastersheet = mastersheet.sort_values("sub_id")
    mastersheet.to_csv(mastersheet_path, index=False)
    print(f"✅ Saved mastersheet to {mastersheet_path}")
    
    return mastersheet

def write_mastersheet_mesa(config, mastersheet_path):
    records = []
    site =f"mesa-sleep" 
    session = 1
    edf_folder = Path(config.paths.input) / "edfs"
    annot_folder = Path(config.paths.input) / "annotations-events-profusion"

    if not edf_folder.is_dir():
        print("[ERROR] No edfs folder.")
    if not annot_folder.is_dir():
        print("[ERROR] No annotation events profusion folder.")
    
    for edf_path in Path(edf_folder).iterdir():
        sub_id = edf_path.stem
        annot_file = f"{sub_id}-profusion.xml"
        annot_path = Path(annot_folder) / annot_file
        
        # Check files exist
        if edf_path is None:
            print(f"[WARNING] Missing EDF for sub-{sub_id}_ses{session}")
            continue
        if annot_path is None:
            print(f"[WARNING] Missing Annotations for sub-{sub_id}_ses{session}")
        
        try: 
            record = write_mastersheet_sub(config, sub_id, site, edf_path, annot_path, session)
        except Exception as e:
            print(f"[ERROR] Cannot read {edf_path.name}: {e}")
            continue   

        records.append(record)
        
    # Convert to DataFrame
    mastersheet = pd.DataFrame(records)
    mastersheet = mastersheet.sort_values("sub_id")
    mastersheet.to_csv(mastersheet_path, index=False)
    print(f"✅ Saved mastersheet to {mastersheet_path}")
    
    return mastersheet
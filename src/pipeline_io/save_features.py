import pandas as pd
from pathlib import Path

def save_features(config, row, extracted_features):
    sub_id = row["sub_id"]
    session = row["session"]

    start_time = row["start_time"]
    if isinstance(start_time, str):
        start_time = pd.to_datetime(start_time)
    measurement_date = start_time.strftime("%Y-%m-%d")
    measurement_datetime = start_time.strftime("%Y-%m-%d %H:%M:%S")

    features = []
    for name, value in extracted_features.items():
        features.append({
            "person_id": sub_id,
            "measurement_date": measurement_date,
            "measurement_datetime": measurement_datetime,
            "measurement_source_value": name,
            "value_as_number": value
        })
    df_new = pd.DataFrame(features)

    sub_key = f"{config.dataset.name.lower()}_ses-{session}_sub-{sub_id}"
    results_path = Path(config.paths.extracted_features) / f"{sub_key}_extracted_features.csv"

    if results_path.exists() and not config.output.overwrite_old_features:
        df_old = pd.read_csv(results_path)
        # Remove any rows with same measurement_source_value to avoid duplicates
        df_old = df_old[~df_old["measurement_source_value"].isin(df_new["measurement_source_value"])]
        df_new = pd.concat([df_old, df_new], ignore_index=True)

    df_new.to_csv(results_path, index=False)
    if config.run.verbose:
        print(f"Saved (or updated) OMOP features for subject {sub_id} → {results_path}")
    
def save_features_wide(config, row, extracted_features):
    sub_id = row["sub_id"]
    session = row["session"]
    flat_row = {"sub_id": sub_id, "session": session}
    flat_row.update(extracted_features)
    df_new = pd.DataFrame([flat_row])

    sub_key = f"{config.dataset.name.lower()}_ses-{session}_sub-{sub_id}"
    results_path = Path(config.paths.extracted_features) / f"{sub_key}_extracted_features_wide.csv"

    if results_path.exists() and not config.output.overwrite_old_features:
        df_old = pd.read_csv(results_path)

        if sub_id in df_old["sub_id"].values:
            existing_row = df_old[df_old["sub_id"] == sub_id].iloc[0].to_dict()

            # Merge: preserve old values unless overwritten
            merged_row = {**existing_row, **flat_row}
            df_old = df_old[df_old["sub_id"] != sub_id]
            df_new = pd.DataFrame([merged_row])
            df_new = pd.concat([df_old, df_new], ignore_index=True)
        else:
            df_new = pd.concat([df_old, df_new], ignore_index=True)

    df_new.to_csv(results_path, index=False)
    if config.run.verbose:
        print(f"Saved (or updated) wide features for subject {sub_id} → {results_path}")
    




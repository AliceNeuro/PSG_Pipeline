import os
import pandas as pd

# === CONFIG ===
results_path = "/wynton/group/andrews/data/PSG_Pipeline_Outputs/extracted_features/mros_ses-1/run_20251023"
print(f"[INFO] Using results path: {results_path}")

output_final = os.path.join(results_path, "mros_ses-1_extracted_features_all.csv")
sub_id_col = "sub_id"

# === GET FILE LIST ===
all_files = [f for f in os.listdir(results_path) if f.endswith("_wide.csv")]
total_files = len(all_files)

print(f"[INFO] Found {total_files} '_wide.csv' files")

if total_files == 0:
    raise FileNotFoundError(f"No '_wide.csv' files found in {results_path}")

# === COMBINE ALL FILES ===
dfs = []
for file_name in all_files:
    file_path = os.path.join(results_path, file_name)
    try:
        df = pd.read_csv(file_path)
        dfs.append(df)
    except Exception as e:
        print(f"[WARN] Could not read {file_path}: {e}")

if not dfs:
    raise RuntimeError("[ERROR] No valid CSVs could be read.")

final_df = pd.concat(dfs, axis=0, ignore_index=True)

if sub_id_col in final_df.columns:
    final_df = final_df.sort_values(by=sub_id_col).reset_index(drop=True)

# === SAVE FINAL COMBINED CSV ===
final_df.to_csv(output_final, index=False)
print(f"[INFO] Final combined CSV saved: {output_final}")
print(f"[INFO] Number of subjects: {len(final_df)}")

# === SPLIT BY STAGE TYPE ===
stage_types = ["WN", "REM", "N2N3"]

for stage in stage_types:
    stage_cols = [col for col in final_df.columns if col.endswith(f"@{stage}")]
    if not stage_cols:
        print(f"[WARN] No columns found for stage '{stage}', skipping.")
        continue

    stage_df = final_df[[sub_id_col] + stage_cols]
    stage_df = stage_df.sort_values(by=sub_id_col).reset_index(drop=True)

    stage_csv_path = os.path.join(results_path, f"mros_ses-1_extracted_features_{stage}.csv")
    stage_df.to_csv(stage_csv_path, index=False)
    print(f"[INFO] Saved {stage_csv_path} with {stage_df.shape[1]-1} stage columns")
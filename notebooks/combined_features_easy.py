import os
import pandas as pd

# === CONFIG ===
dataset_name = "shhs_ses-1" 
run_date = "20260202"
#feature = "aasm"
results_path = f"/wynton/group/andrews/data/PSG_Pipeline_Outputs/extracted_features/{dataset_name}/run_{run_date}/"
print(results_path)
output_final = os.path.join(results_path, f"{dataset_name}_all.csv")
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
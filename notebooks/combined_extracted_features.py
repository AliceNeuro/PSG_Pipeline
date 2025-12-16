import os
import pandas as pd
import math

# === CONFIG ===
dataset_name = "shhs_ses-1" 
run_date = "20251215"
feature = "vb"
results_path = f"/wynton/group/andrews/data/PSG_Pipeline_Outputs/extracted_features/{dataset_name}/run_{run_date}_{feature}/"
print(results_path)
# results_path = "/wynton/group/andrews/data/PSG_Pipeline_Outputs/extracted_features/shhs_ses-1/run_20251013"
output_final = os.path.join(results_path, f"{dataset_name}_{feature}_all.csv")
sub_id_col = "sub_id"
final_df = None

batch_size = 250  # number of CSVs per batch

tmp_dir = os.path.join(results_path, "_tmp_batches")
if not os.path.exists(tmp_dir):
    os.mkdir(tmp_dir)
    
# === GET FILE LIST ===
all_files = os.listdir(results_path)
total_files = len(all_files)
num_batches = math.ceil(total_files / batch_size)

print(f"[INFO] Found {total_files} files → {num_batches} batches of {batch_size}")


# === PROCESS BY BATCH ===
for i in range(num_batches):
    start = i * batch_size
    end = min(start + batch_size, total_files)
    batch_files = all_files[start:end]
    print(f"[INFO] Processing batch {i+1}/{num_batches} ({len(batch_files)} files)")

    dfs = []
    for file_name in batch_files:
        if not file_name.endswith("_wide.csv"):
            continue
        file_path = os.path.join(results_path, file_name)
        try:
            df = pd.read_csv(file_path)
            dfs.append(df)
        except Exception as e:
            print(f"[WARN] Could not read {file_path}: {e}")

    if dfs:
        combined_df = pd.concat(dfs, axis=0, ignore_index=True)
        if sub_id_col in combined_df.columns:
            combined_df = combined_df.sort_values(by=sub_id_col).reset_index(drop=True)

        # --- Save batch output ---

        batch_output = os.path.join(tmp_dir, f"batch_{i+1:03d}.csv")
        combined_df.to_csv(batch_output, index=False)
    else:
        print("[WARN] No valid CSVs in this batch.")

# === COMBINE ALL BATCHES INTO FINAL CSV ===
batch_files = sorted(os.listdir(tmp_dir))
dfs = []
for f in batch_files:
    batch_path = os.path.join(tmp_dir, f)
    try:
        dfs.append(pd.read_csv(batch_path))
    except Exception as e:
        print(f"[WARN] Could not read batch file {batch_path}: {e}")

if dfs:
    final_df = pd.concat(dfs, ignore_index=True)
    final_df = final_df.sort_values(by=sub_id_col).reset_index(drop=True)
    final_df.to_csv(output_final, index=False)
    print(f"[INFO] Final combined CSV saved: {output_final}")

# Load combined file
if final_df is None:
    final_df = pd.read_csv(output_final)
    print(f"[INFO] Number of row in combined all is {len(final_df)}")

# Identify stage types
stage_types = ["WN", "REM", "N2N3"]

# Split columns for each stage type, always include 'sub_id'
for stage in stage_types:
    # Select columns: sub_id + columns ending with f"@{stage}"
    stage_cols = [col for col in final_df.columns if col.endswith(f"@{stage}")]
    stage_cols = [sub_id_col] + stage_cols
    
    stage_df = final_df[stage_cols]
    
    # Sort by sub_id
    stage_df = stage_df.sort_values(by=sub_id_col).reset_index(drop=True)
    
    # Save to CSV
    stage_csv_path = os.path.join(results_path, f"{dataset_name}_{feature}_{stage}.csv")
    stage_df.to_csv(stage_csv_path, index=False)
    print(f"[INFO] Saved {stage_csv_path} with {stage_df.shape[1]-1} stage columns")
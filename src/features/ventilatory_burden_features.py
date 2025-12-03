import os 
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
import scipy.io as sio
from scipy.signal import resample
from scipy.stats import mode

def extract_vb(row, tmp_dir_sub, resp_data, full_sleep_stages, verbose):
    psg_id = f"sub-{row['sub_id']}_ses-{row['session']}"
    if not resp_data:
        if verbose:
            print(f"{psg_id}: No respiratory data found → skipping")
        return None

    # Resample if below 20 Hz
    sfreq_resp = float(resp_data['sfreq_signal'])
    if sfreq_resp < 20.0: 
        sfreq_resp = 20.0

    # Downsample resp to match sfreq_global
    sfreq_global = float(row['sfreq_global'])
    ratio = sfreq_global / sfreq_resp
    if np.isclose(ratio, round(ratio)):
        factor = int(round(ratio))
        full_resp = resp_data['full_signal'][::factor]
        full_sleep_stages = full_sleep_stages[::factor]
    else:
        # Compute new number of samples
        n_samples = int(round(len(resp_data['full_signal']) * sfreq_resp / sfreq_global))
        full_resp = resample(resp_data['full_signal'], n_samples)
        
        # Resample sleep stages by nearest-neighbor
        idx = np.linspace(0, len(full_sleep_stages) - 1, n_samples).astype(int)
        full_sleep_stages = full_sleep_stages[idx]
    
    # Geat df_breath
    breath_mat = tmp_dir_sub / f"breath_{psg_id}.mat"
    df_breath = get_breath_array(breath_mat, full_resp, sfreq_resp, verbose)
    
    # Reduce df_breath 
    sleep_ids = np.where(np.isin(full_sleep_stages, [1, 2, 3, 4]))[0]
    start_index = sleep_ids[0]
    end_index = sleep_ids[-1] + 1 # include the last index
    df_breath = df_breath[(df_breath['insp_onset'] >= start_index) & (df_breath['exp_offset'] <= end_index)]

    # Add the most frequent sleep stage per breath
    sleep_stage_per_breath = []
    for idx, row in df_breath.iterrows():
        start_idx = int(row['insp_onset'])
        stop_idx = int(row['exp_offset']) + 1
        stages_slice = full_sleep_stages[start_idx:stop_idx]
        valid_stages = stages_slice[~np.isnan(stages_slice)]
        
        if len(valid_stages) == 0:
            if verbose:
                print(f"[WARNING] {psg_id}: No valid stages for breath idx:", idx, "Start:", start_idx, "Stop:", stop_idx)
            sleep_stage_per_breath.append(np.nan)
        else: # find the most frequent stage
            most_common_stage = mode(valid_stages, keepdims=True).mode[0]
            sleep_stage_per_breath.append(most_common_stage)
            
    df_breath['sleep_stage'] = sleep_stage_per_breath

    # Results per sleep stages
    results = {}
    sleep_periods = ['WN', 'SLEEP', 'NREM', 'N2N3', 'REM']

    for period in sleep_periods:
        if period == 'WN':
            df_period = df_breath  # all breaths
        elif period == 'SLEEP':
            df_period = df_breath[df_breath['sleep_stage'].isin([1,2,3,4])]
        elif period == 'NREM':
            df_period = df_breath[df_breath['sleep_stage'].isin([1,2,3])]
        elif period == 'N2N3':
            df_period = df_breath[df_breath['sleep_stage'].isin([1,2])]
        elif period == 'REM':
            df_period = df_breath[df_breath['sleep_stage'].isin([4])]

        # Only compute if enough breaths
        if len(df_period) >= 5:
            amp = df_period['normalized_amplitude']
            amp = np.clip(amp, 0, 200)
            bins = np.arange(0, 205, 5)
            hist, _ = np.histogram(amp, bins=bins)
            hist_percentage = (hist/len(amp)) * 100
            results[f'vb@{period}'] = np.sum(hist_percentage[:10]) # <= 50% 
        else:
            results[f'vb@{period}'] = np.nan

    return results

def get_breath_array(breath_mat, full_resp, sfreq_resp, verbose):
    sio.savemat(breath_mat, {'nas_pres': full_resp, 
                            'fs': sfreq_resp, 
                            'opts': {'plotFig': 0}})

    project_src_root = Path(__file__).resolve().parents[1] 
    mat_breath_script = project_src_root / "external_tools/matlab/breath_table/call_breathtable.m"

    # Build MATLAB command
    command = [
        "/usr/local/matlab/R2024a/bin/matlab", # "/usr/local/matlab/R2024a/bin/matlab", # "/Applications/MATLAB_R2025a.app/bin/matlab"
        "-nojvm",
        "-nosplash",
        "-nodesktop",
        "-softwareopengl",
        "-batch",
        f'addpath("{mat_breath_script.parent}"); call_breathtable("{breath_mat}")'
    ]

    subprocess.run(
                command,
                check=True,
                stdout=None if verbose else subprocess.DEVNULL,
                stderr=None if verbose else subprocess.DEVNULL
            )

    # Load output from MATLAB
    breath_data = sio.loadmat(breath_mat)
    breath_array = breath_data['breath_array']

    colnames = ['breath_id', 'insp_onset', 'peak', 'insp_breath_offset',
            'exp_peak', 'exp_onset', 'exp_offset', 'fs',
            'RespiratoryRate', 'val_fl', 'Ttot', 'normalized_amplitude']

    df_breath = pd.DataFrame(breath_array, columns=colnames)

    return df_breath
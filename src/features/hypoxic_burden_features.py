import os
import h5py
import mne
import neurokit2 as nk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import filtfilt, find_peaks, resample


# Main extraction def
def extract_hb(row, spo2_data, df_events, full_sleep_stages, verbose):
    # Read inputs
    psg_id = f"sub-{row['sub_id']}_ses-{row['session']}"
    if not spo2_data:
        if verbose:
            print(f"{psg_id}: No SpO2 data found → skipping")
        return None
    
    # Downsample resp to match sfreq_global
    sfreq_global = float(row['sfreq_global'])
    sfreq_spo2  = spo2_data['sfreq_signal']
    ratio = sfreq_global / sfreq_spo2
    if np.isclose(ratio, round(ratio)):
        factor = int(round(ratio))
        spo2_signal = spo2_data['full_signal'][::factor]
        full_sleep_stages = full_sleep_stages[::factor]
    else:
        # Compute new number of samples
        n_samples = int(round(len(spo2_data['full_signal']) * sfreq_spo2 / sfreq_global))
        spo2_signal = resample(spo2_data['full_signal'], n_samples)
        
        # Resample sleep stages by nearest-neighbor
        idx = np.linspace(0, len(full_sleep_stages) - 1, n_samples).astype(int)
        full_sleep_stages = full_sleep_stages[idx]

    # Extract apnea events (end time of each event)
    df_apnea = (
        df_events[df_events["event_type"].astype(str).str.contains("pnea", case=False, na=False)]
        .copy()
        .reset_index(drop=True)
    )
    df_apnea["end_time"] = df_apnea["onset"].astype(float) + df_apnea["duration"].astype(float)
    apnea_event_times = np.array(df_apnea["end_time"])
    
    # If no apnea events or less than 2 → mark apnea HB as NaN
    if len(apnea_event_times) < 2:
        if verbose:
            print(f"{psg_id}: Not enough apnea events so apnea HB set to NaN")
        HB_apnea = (np.nan, np.nan, np.nan)
    else: # Compute HB apnea
        df_apnea_hb = calc_hypoxic_burden(apnea_event_times, spo2_signal, sfreq_spo2)
        HB_apnea = hb_per_stages(df_apnea_hb, full_sleep_stages, sfreq_spo2)

    HB_per_hour_apnea, HB_NREM_per_hour_apnea, HB_REM_per_hour_apnea = HB_apnea


    # Extract desaturation events (mid time of each event)
    df_desat = (
        df_events[
            df_events["event_type"].astype(str).str.contains("desat", case=False, na=False)
            & ~df_events["event_type"].astype(str).str.contains("artifac", case=False, na=False)
        ]
        .copy()
        .reset_index(drop=True)
    )
    if len(df_desat) == 0: # If no desat events found → detect automatically
        df_desat = detect_oxygen_desaturation(spo2_signal, is_plot=False)
        
    df_desat["mid_time"] = df_desat["onset"].astype(float) + (df_desat["duration"].astype(float)/2)
    desat_event_times = np.array(df_desat["mid_time"])

    # If no desats or <2 → mark desat HB as NaN
    if len(desat_event_times) < 2:
        if verbose:
            print(f"{psg_id}: Not enough desaturation events → desat HB set to NaN")
        HB_desat = (np.nan, np.nan, np.nan)
    else:
        df_desat_hb = calc_hypoxic_burden(desat_event_times, spo2_signal, sfreq_spo2)
        HB_desat = hb_per_stages(df_desat_hb, full_sleep_stages, sfreq_spo2)

    HB_per_hour_desat, HB_NREM_per_hour_desat, HB_REM_per_hour_desat = HB_desat

    results = {
        "hb_per_hour_desat@WN": HB_per_hour_desat,
        "hb_per_hour_desat@NREM": HB_NREM_per_hour_desat,
        "hb_per_hour_desat@REM": HB_REM_per_hour_desat,
        "hb_per_hour_apnea@WN": HB_per_hour_apnea,
        "hb_per_hour_apnea@NREM": HB_NREM_per_hour_apnea,
        "hb_per_hour_apnea@REM": HB_REM_per_hour_apnea,
    }
    return results



### Useful code
B = [0.000109398212241, 0.000514594526374, 0.001350397179936, 0.002341700062534,
     0.002485940327008, 0.000207543145171, -0.005659450344228, -0.014258087808069,
     -0.021415481383353, -0.019969417749860, -0.002425120103463, 0.034794452821365,
     0.087695691366900, 0.144171828095816, 0.187717212244959, 0.204101948813338,
     0.187717212244959, 0.144171828095816, 0.087695691366900, 0.034794452821365,
     -0.002425120103463, -0.019969417749860, -0.021415481383353, -0.014258087808069,
     -0.005659450344228, 0.000207543145171, 0.002485940327008, 0.002341700062534,
     0.001350397179936, 0.000514594526374, 0.000109398212241]

BAD_SPO2_THRESHOLD = 80

def filter_spo2(spo2_arr, spo2_sfreq, event_end_time, verbose=False, time_span=120):
    # Replace abnormal values with the mean
    spo2_mean = np.mean(spo2_arr[spo2_arr >= BAD_SPO2_THRESHOLD])
    spo2_arr[spo2_arr < BAD_SPO2_THRESHOLD] = spo2_mean

    if spo2_sfreq != 1:
        spo2_arr = nk.signal_resample(spo2_arr, sampling_rate=spo2_sfreq, desired_sampling_rate=1)
        spo2_sfreq = 1

    # Reduce SpO₂ jitter and adjust SpO₂ resolution to 0.5
    spo2_filtered = filtfilt(B, 1, spo2_arr, axis=0, padtype='odd')
    spo2_filtered *= 2
    spo2_filtered = np.round(spo2_filtered) / 2

    return spo2_filtered

    
def calc_hypoxic_burden(event_times, spo2_arr, sfreq_spo2, verbose=False, time_span=120):
    # Assume the duration of a respiratory event is 10–120 s; the maximum delay of hypoxemia caused by a respiratory event is 120 s
    all_ah_related_spo2 = []
    good_event_ids = []
    for ei, et in enumerate(event_times):
        start_idx = int((et - time_span) * sfreq_spo2)
        end_idx = int((et + time_span) * sfreq_spo2)
        nearby_spo2 = spo2_arr[start_idx:end_idx]
        if len(nearby_spo2) < 2*time_span*sfreq_spo2 \
                or np.mean(nearby_spo2 < BAD_SPO2_THRESHOLD)>0.3:
            continue
        filtered_spo2 = filter_spo2(nearby_spo2, sfreq_spo2, et, verbose, time_span)
        assert sfreq_spo2 == 1, f"Unexpected SpO₂ frequency: {sfreq_spo2}"
        all_ah_related_spo2.append(filtered_spo2)
        good_event_ids.append(ei)
    
    # Get average drop (start and end on average curve)
    all_spo2_dest = np.array(all_ah_related_spo2)
    avg_spo2 = all_spo2_dest.mean(axis=0)
    avg_spo2 = filtfilt(B, 1, avg_spo2, axis=0, padtype='odd')
    peaks, _ = find_peaks(avg_spo2)
    start_secs = peaks[np.where(peaks < time_span)[0][-1]]
    end_secs = peaks[np.where(peaks > time_span)[0][0]]

    burdens = []
    for spo2_dest_curve in all_spo2_dest:
        baseline_spo2 = np.max(spo2_dest_curve[time_span - 100:time_span])
        interest_spo2 = spo2_dest_curve[start_secs: end_secs]
        burdens.append( sum(baseline_spo2 - interest_spo2)/60 )

    res = pd.DataFrame(data={'EventTime':event_times})
    res.loc[good_event_ids, 'HB'] = burdens
    return res


def hb_per_stages(df_hb, full_sleep_stages, sfreq_spo2):
    # Add Sleep Stages 
    full_sleep_stages_sec = full_sleep_stages[::int(sfreq_spo2)]
    ids = np.clip(df_hb['EventTime'].astype(int).values, 0, len(full_sleep_stages_sec)-1)
    df_hb['Stage'] = full_sleep_stages_sec[ids]

    # Start and end sleep in sec
    sleep_ids = np.where(np.isin(full_sleep_stages, [1, 2, 3, 4]))[0]
    sleep_start = int(sleep_ids[0] / sfreq_spo2)
    sleep_end = int((sleep_ids[-1] + 1) / sfreq_spo2)

    # Total HB during sleep (per hour)
    total_HB = df_hb['HB'][(df_hb['EventTime'] >= sleep_start) & (df_hb['EventTime'] < sleep_end)].sum()
    total_sleep_hours = (sleep_end - sleep_start) / 3600
    HB_per_hour = total_HB / total_sleep_hours

    # NREM hypoxic burden
    mask_nrem = np.in1d(df_hb['Stage'], [1,2,3])
    if mask_nrem.sum() > 0:
        total_nrem_HB = df_hb['HB'][mask_nrem].sum()
        nrem_hours = np.in1d(full_sleep_stages_sec[int(sleep_start):int(sleep_end)], [1,2,3]).sum() / 3600
        HB_NREM_per_hour = total_nrem_HB / nrem_hours
    else:
        HB_NREM_per_hour = 0

    # REM hypoxic burden
    mask_rem = np.in1d(df_hb['Stage'], [4])
    if mask_rem.sum() > 0:
        total_rem_HB = df_hb['HB'][mask_rem].sum()
        rem_hours = np.in1d(full_sleep_stages_sec[int(sleep_start):int(sleep_end)], [4]).sum() / 3600
        HB_REM_per_hour = total_rem_HB / rem_hours
    else:
        HB_REM_per_hour = 0
    
    return HB_per_hour, HB_NREM_per_hour, HB_REM_per_hour


def detect_oxygen_desaturation(spo2, is_plot=False, duration_max=120, return_type='pd'):
    spo2_max = spo2[0]  # Initialize maximum SpO2 value
    spo2_max_index = 1  # Initialize index of maximum SpO2 value
    spo2_min = 100  # Initialize minimum SpO2 value
    des_onset_pred_set = np.array([], dtype=int)  # Collection of predicted desaturation onset points
    des_duration_pred_set = np.array([], dtype=int)  # Collection of predicted desaturation durations
    des_level_set = np.array([])  # Collection of recorded desaturation events (e.g., 2%, 3%, 4%, 5% drops, etc.)
    des_onset_pred_point = 0  # Predicted onset point of the current desaturation event
    des_flag = 0  # Flag indicating whether a desaturation event is occurring
    ma_flag = 0  # Flag indicating whether a motion artifact event is occurring
    spo2_des_min_thre = 2  # Minimum desaturation threshold (in %) to trigger detection
    spo2_des_max_thre = 50  # Motion artifact threshold (if SpO2 drops more than 50%, it's likely an artifact)
    duration_min = 5  # Minimum duration (in seconds) for a desaturation event to be recorded
    prob_end = []  # List to store probable end points of desaturation events

    for i, current_value in enumerate(spo2):

        des_percent = spo2_max - current_value  # Desaturation value

        # Detect motion artifacts
        if ma_flag and (des_percent < spo2_des_max_thre):
            if des_flag and len(prob_end) != 0:
                des_onset_pred_set = np.append(des_onset_pred_set, des_onset_pred_point)
                des_duration_pred_set = np.append(des_duration_pred_set, prob_end[-1] - des_onset_pred_point)
                des_level_point = spo2_max - spo2_min
                des_level_set = np.append(des_level_set, des_level_point)
            # Reset
            spo2_max = current_value
            spo2_max_index = i
            ma_flag = 0
            des_flag = 0
            spo2_min = 100
            prob_end = []
            continue

        # If desaturation value is greater than 2%, record the onset time
        if des_percent >= spo2_des_min_thre:
            if des_percent > spo2_des_max_thre:
                ma_flag = 1
            else:
                des_onset_pred_point = spo2_max_index
                des_flag = 1
                if current_value < spo2_min:
                    spo2_min = current_value

        if current_value >= spo2_max and not des_flag:
            spo2_max = current_value
            spo2_max_index = i

        elif des_flag:

            if current_value > spo2_min:
                if current_value > spo2[i - 1]:
                    prob_end.append(i)

                # Locate consecutive SpO2 drop points
                if current_value <= spo2[i - 1] < spo2[i - 2]:
                    spo2_des_duration = prob_end[-1] - spo2_max_index

                    # If the drop duration is too short, it is not considered a desaturation event
                    if spo2_des_duration < duration_min:
                        spo2_max = spo2[i - 2]
                        spo2_max_index = i - 2
                        spo2_min = 100
                        des_flag = 0
                        prob_end = []
                        continue

                    else:
                        # If the drop duration meets the requirement, record this desaturation event
                        if duration_min <= spo2_des_duration <= duration_max:
                            des_onset_pred_set = np.append(des_onset_pred_set, des_onset_pred_point)
                            des_duration_pred_set = np.append(des_duration_pred_set, spo2_des_duration)
                            des_level_point = spo2_max - spo2_min
                            des_level_set = np.append(des_level_set, des_level_point)

                        # If the drop duration is too long, it indicates multiple desaturation events that need to be recorded separately
                        else:
                            # Record the first desaturation event
                            des_onset_pred_set = np.append(des_onset_pred_set, des_onset_pred_point)
                            des_duration_pred_set = np.append(des_duration_pred_set, prob_end[0] - des_onset_pred_point)
                            des_level_point = spo2_max - spo2_min
                            des_level_set = np.append(des_level_set, des_level_point)

                            # Recheck for possible desaturation events
                            remain_spo2 = spo2[prob_end[0]:i + 1]
                            _onset, _duration, _des_level = detect_oxygen_desaturation(remain_spo2, is_plot=False, return_type='tuple')
                            des_onset_pred_set = np.append(des_onset_pred_set, _onset + prob_end[0])
                            des_duration_pred_set = np.append(des_duration_pred_set, _duration)
                            des_level_set = np.append(des_level_set, _des_level)

                        spo2_max = spo2[i - 2]
                        spo2_max_index = i - 2
                        spo2_min = 100
                        des_flag = 0
                        prob_end = []

    return pd.DataFrame(data={'onset':des_onset_pred_set, 'duration':des_duration_pred_set, 'desaturation':des_level_set})
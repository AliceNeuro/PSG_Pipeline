import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone


def process_sleep_stages(full_sleep_stages, sfreq_global, start_time, verbose):
    # Keep only sleep
    sleep_ids = np.where(np.isin(full_sleep_stages, [1, 2, 3, 4]))[0]
    start_index = sleep_ids[0]
    end_index = sleep_ids[-1] + 1 # include the last index
    sleep_stages = full_sleep_stages[start_index:end_index] # in samples

    # Adjust the sleep time 
    sleep_onset_offset_sec = float(start_index / sfreq_global)
    start_time = datetime.fromisoformat(start_time)
    start_time = start_time.replace(tzinfo=timezone.utc)
    sleep_onset_time = start_time + timedelta(seconds=sleep_onset_offset_sec)
    
    if verbose: 
        print("Start time:", start_time)
        print("Sleep Onset seconds:", sleep_onset_offset_sec)
        print("Sleep Onset time:", sleep_onset_time)

    return sleep_stages, sleep_onset_time
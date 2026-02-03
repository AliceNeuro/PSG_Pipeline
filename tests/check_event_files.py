import os
import pandas as pd
import numpy as np


TWELVE_HOURS_SEC = 20 * 60 * 60  # 72000 seconds
event_folder = "/wynton/group/andrews/data/PSG_Pipeline_Outputs/events/hsp_bidmc/"

#print(os.listdir(event_folder)[:2])

for file in sorted(os.listdir(event_folder))[:5]:
    if file.endswith(".csv"):  
        file_path = os.path.join(event_folder, file)
        df = pd.read_csv(file_path)

        if "onset" not in df.columns:
            print(f"{file}: missing 'onset' column")
            continue

        onset = df["onset"].values

        # Check if onset is ordered
        onset_clean = onset[~np.isnan(onset)] # Remove NaNs first
        if not np.all(np.diff(onset_clean) >= 0):
            print(f"{file}: onset not in order")

        # Check for events beyond 20 hours
        weird_events = onset[onset > TWELVE_HOURS_SEC]
        if len(weird_events) > 0:
            print(f"{file}: {len(weird_events)} events beyond 20 hours detected")
            print(weird_events)


import os
import pandas as pd

# Get the list of files to keep 
master = pd.read_csv("/wynton/group/andrews/data/PSG_Pipeline_Outputs/mastersheets/hsp_mgb_mastersheet.csv")
master["h5_path"] = master["h5_path"].str.split('/').str[-1]
keep_files = set(master["h5_path"].to_list())

# Go though all the files computed
path_h5 = "/wynton/group/andrews/data/PSG_Pipeline_Outputs/h5_data/hsp_mgb/"
for file in os.listdir(path_h5):
    if file not in keep_files:
        print(f"Deleting: {file}")
        os.remove(os.path.join(path_h5, file)) 
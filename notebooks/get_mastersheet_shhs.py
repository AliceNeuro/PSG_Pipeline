from pathlib import Path
import os
import sys
import mne

project_root = Path.cwd().parent
src_path = project_root / "src"
sys.path.append(str(src_path))
os.chdir(project_root)
sys.argv = ["notebook", "--config", "/wynton/home/leng/alice-albrecht/PSG_Pipeline/config/shhs_ses-2_config.yaml"] 

from config.read_config import read_config
from pipeline_io.get_mastersheet import get_mastersheet

config, config_path = read_config()
print(config.dataset.name, config.dataset.session)
mastersheet = get_mastersheet(config)
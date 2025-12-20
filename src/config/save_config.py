from pathlib import Path
import shutil
from datetime import datetime, timedelta

def save_config(config, config_path):
    """
    Copy the original config YAML to a timestamped folder.
    
    Args:
        config_path: Path to original YAML config
        save_folder: Path to output folder where copy should be stored
    """
    config_path = Path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Timestamp folder
    #date_time = datetime.now().strftime("%Y%m%d")
    date_time = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
    save_folder = Path(config.paths.extracted_features) / f"run_{date_time}" 
    print("Saving extracted features in:", save_folder)
    config.paths.extracted_features = save_folder
    save_folder.mkdir(parents=True, exist_ok=True)
    save_path = save_folder / f"run_{date_time}_{config_path.name}"


    # Copy config
    shutil.copy(config_path, save_path)

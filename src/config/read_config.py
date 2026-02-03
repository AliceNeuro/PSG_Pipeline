import argparse
from pathlib import Path
from box import Box
import yaml

def read_config():
    # --- Read config file ---
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config YAML")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_file():
        raise FileNotFoundError(f"[ERROR] Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = Box(yaml.safe_load(f))

    # --- Check input exists ---
    input_path = Path(config.paths.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    
    # --- Create folders ---
    output_path = Path(config.paths.output)
    output_path.mkdir(parents=True, exist_ok=True)

    mastersheet_folder = output_path / "mastersheets"
    mastersheet_folder.mkdir(parents=True, exist_ok=True)
    output_paths = {"mastersheets": mastersheet_folder}
    
    output_folder_names = ["h5_data", "events", "extracted_features"]
    dataset_session = (
        f"{config.dataset.name.lower()}_ses-{config.dataset.session}"
        if config.dataset.session is not None
        else config.dataset.name.lower()
    )

    for folder in output_folder_names:
        output_folder = output_path / folder
        output_folder.mkdir(parents=True, exist_ok=True)
        subfolder = output_folder / dataset_session
        subfolder.mkdir(parents=True, exist_ok=True)
        output_paths[folder] = subfolder

    # --- Attach to config.paths ---
    for name, path in output_paths.items():
        setattr(config.paths, name, path)

    return config, config_path
    
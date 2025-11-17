# PSG Sleep Feature Extraction Pipeline

This repository provides a modular, extensible pipeline to extract sleep-related features from polysomnography (PSG) data. It supports multiple datasets, cohorts, and visit numbers — whether raw or BIDS-formatted.

---

## 🌟 Features

Supports computation of sleep-related metrics including:

- **Cardiovascular**
  - Heart Rate Variability (HRV)
  - Heart Rate Nadir (HRnadir)
  - Cardiopulmonary Coupling (CPC)

- **Respiratory (TO DO)** 
  - Arousal Burden (AB)
  - Ventilatory Burden (VB)
  - Hypoxia Burden (HB)

- **Neuro / EEG (TO DO)**
  - EEG Microstructure (EEGmicrostructures)
  - Brain Age Index (BAI)

---

## 🗂️ Dataset Organization (Local, External to Git Repo)

### Inputs 
Datasets are not stored in the repository. Each dataset/session has its own config file, where the input paths can be customized. If you are working with a new dataset/session that has a specific folder structure, you can define your own write_mastersheet function in `src/pipeline_io/get_mastersheet.py` and register it in `MASTERSHEET_WRITERS`.

```
MASTERSHEET_WRITERS = {
        "mros_ses-1": write_mastersheet_mros_ses1,
        "mros_ses-2": write_mastersheet_mros_ses2,
        "hsp_ses-1": write_mastersheet_hsp_ses1,
    }
```

### Outputs
All pipeline outputs are stored in the output paths defined in the dataset/session config file. The outputs are organized into several types of files:
1. Mastersheets – aggregated subject metadata.
2. Intermediate HDF5 files (.h5) – generated from the original EDF signals.
3. Event files – containing all annotations such as sleep stages, arousals, and flow events.
4. Extracted features – computed metrics from the signals (e.g., HRV, CPC).

```
PSG_Pipeline_Outputs/
├── mastersheets/
│   ├── mros_ses-1_mastersheet.csv
│   ├── mros_ses-2_mastersheet.csv
│   └── hsp_ses-1_mastersheet.csv
├── h5_data/
│   ├── mros_ses-1/
│   │   ├── mros_ses-1_sub-sd8001_signals.h5
│   │   └── ...
│   ├── mros_ses-2/
│   └── hsp_ses-1/
├── events/
│   ├── mros_ses-1/
│   │   ├── mros_ses-1_sub-sd8001_events.csv
│   │   └── ...
│   ├── mros_ses-2/
│   └── hsp_ses-1/
└── extracted_features/
    ├── mros_ses-1/
    │   ├── mros_ses-1_sub-sd8001_extracted_features.csv
    │   └── ...
    ├── mros_ses-2/
    └── hsp_ses-1/
```

---

## 📁 PSG Pipeline Structure

```
PSG-PIPELINE/
├── config/                # YAML config files per dataset
│   ├── hsp_ses-1_config.yaml
│   └── mros_ses-1_config.yaml
│
├── external_tools/        # External dependencies (e.g., MATLAB or C code)
│   ├── c_modules/
│   └── matlab/
│
├── notebooks/             # Notebooks for exploration and debugging
│
├── docs/                  # More detailed documentations and project notes
│
├── src/                   # Core processing logic
│   ├── analysis/          # Full analysis from h5 to extracted features
│   ├── config/            # Reading the config file 
│   ├── external_tools/    # Matlab, C modules used by the pipeline 
│   ├── features/          # Exrtraction of features
│   ├── pipeline_io/       # Loading, Writing files
│   ├── utils/             # Modality-specific helpers
│   └── main.py/           # Main execution control
│
├── tests/                 # Unit tests
│
├── tmp/                   # Temporary/intermediate processing files
│
├── requirements.txt
├── README.md
├── run_pipeline.py        # Entry point
└── .gitignore          # Ignore datasets, outputs, tmp files, etc.
```

---

## ⚙️ Dependencies & Environment Setup

This project is built with Python 3.9+ and uses several scientific libraries for PSG data processing. You can install dependencies in one of three ways:

**Option 1: Use conda (recommended)**
If you use Anaconda or Miniconda, you can create an isolated environment:
```
conda create -n env-psg-pipeline python=3.9
conda activate env-psg-pipeline
pip install -r requirements.txt
```
Or, if you prefer using an environment YAML file:
```
conda env create -f environment.yml
conda activate env-psg-pipeline
```


**Option 2: Use a venv (alternative)**
Create a virtual environment with Python’s built-in tool:
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Option 3: Install with pip (not recommended)**
If you’re not using a virtual environment:
```
pip install -r requirements.txt
```  

**📝 Notes**  
- **External Tools:** If you plan to use tools in the external_tools/ folder:
  - Make sure MATLAB (or MATLAB Runtime) is installed if required
  - Any C modules must be compiled manually for your environment

- **hrv-analysis Compatibility Fix:** To ensure compatibility, a small modification is needed in the hrv-analysis library:
  - File to edit: hrvanalysis/extract_features.py
  - Change this line:
    ```
    from astropy.stats import LombScargle
    ```
  - To this:
    ```
    from astropy.timeseries import LombScargle
    ``` 
---

## 🚀 Running the Pipeline

Each dataset has its own config file in config/, specifying:
- Dataset input path 
- Output path
- Feature sets to compute

```
python run_pipeline.py --config config/mros_ses-1_config.yaml
```
---

## 🛠️ Toolboxes and External Code

Some processing steps rely on:
- MATLAB (e.g., MrOS-specific R-peak detection)
- C/C++ binaries (e.g., ERD computation)

These tools are stored under version control only if legally redistributable. Compiled binaries should be generated locally and kept outside GitHub.

---

## ✅ Validations

For some datasets, precomputed or reference features (e.g., validated HRV results) are stored for testing purposes — these are not tracked by Git, but used to verify output correctness.

---

## 👩‍💻 Maintainer

Alice Albrecht — Data Specialist in Sleep Research at UCSF   
GitHub: [@AlbrechtAlice](https://github.com/AlbrechtAlice)

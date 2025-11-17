# Feature Extraction Documentation

This document explains how we extract features in the pipeline. It includes the types of features, how we define analysis windows, and why certain choices were made.

---

## 📦 Window Settings (from config)

These parameters control how ECG windows are defined and selected for feature extraction:
1.	Window size and step
    - `window_size_min = 5` → each window is 5 minutes (10 × 30 s epochs)
	- `window_step_epochs = 1` → windows move forward by 30 s
2.	Quality filtering
	- `clean_rpeaks_ratio_threshold = 0.98` → keep only windows with ≥98% clean R-peaks
3.	Sleep stage selection
    - `stage_types = ['WN', 'REM', 'N2N3']` → stages in which features are computed
	- `stage_purity = 0.9` → e.g., for 10 epochs, at least 9 must match the target stage
4.	Overlap control
	- `allowed_overlap_epochs = 1` → allows minimal overlap between windows to ensure independence

These settings ensure that each window is clean, stage-specific, and reliable for feature computation.

---

## ❤️ HRV – Heart Rate Variability

We compute HRV features on each clean window. 

### Extracted HRV Features

```
feat_names_hrv = [
    'mean_nni', 'sdnn', 'sdsd', 'nni_50', 'pnni_50', 'nni_20', 'pnni_20',
    'rmssd', 'median_nni', 'range_nni', 'cvsd', 'cvnni', 'mean_hr', 'max_hr', 'min_hr', 'std_hr',
    'lf', 'hf', 'lf_hf_ratio', 'lfnu', 'hfnu', 'total_power', 'vlf',
    'sd1', 'sd2', 'ratio_sd2_sd1', 'sampen'
]
```

These cover time-domain, frequency-domain, and non-linear measures.

---

## 🌬️ CPC – Cardiopulmonary Coupling

CPC looks at the relationship between breathing and heart rate.

### What is EDR?

EDR = ECG-derived respiration. We extract it from the full ECG signal once for the entire night (not per window). This avoids discontinuities and gives better quality for CPC analysis.

### RRI Handling for CPC

CPC is sensitive to the natural variability in heart rate and how it correlates with respiration. Thus, we intentionally avoid get_nn_intervals() to preserve this variability and not create fake values 

#### Why we avoid get_nn_intervals() for CPC:
- ```get_nn_intervals()``` detects ectopic or artifact-likely RR intervals and replaces them through interpolation.
- We only remove implausibly RR intervals, e.g., > 3s  or < 0.3s(configurable):

```
m_rrini_sec = config.analysis.max_rri_sec  # e.g., 0.3s
max_rri_sec = config.analysis.max_rri_sec  # e.g., 3.0s
```

- If too many intervals are removed (> 1% quantile), we print a warning:

```
if removed_ratio > config.analysis.removal_warning_threshold:
    print(f"[WARNING] Sub {sub_id}: Removed {removed_ratio*100:.1f}% of RRIs ({min_rri_sec:.2f}-{max_rri_sec:.2f}s)")
```

### Alignment, Filtering & Downsampling

To align RR and EDR signals for CPC feature extraction, we perform the following steps:

- Time Alignment
    - Use ```interp1d``` to interpolate RRI and EDR on the ECG timeline -> to be one the same timeline.
	- Critically, we preserve NaNs — we don’t fill gaps introduced by outlier removal. **This avoids fabricating false physiological data.**
	- This ensures that **features are computed only on valid data.**

- Anti-Aliasing Filtering

    Missing values (NaNs) must be filled before resampling to avoid introducing spikes or discontinuities. Linear interpolation is used: internal NaNs are interpolated linearly between neighboring valid points, and NaNs at the edges are filled using the nearest valid value (flat line).
    
    ```
    rri_filled = pd.Series(rri_sec).interpolate(limit_direction="both").to_numpy()
    edr_filled = pd.Series(edr_sec).interpolate(limit_direction="both").to_numpy()
    ```
    When reducing the sampling rate, a low-pass filter is needed to remove high-frequency content above the new Nyquist frequency: `mne.filter.resample` applies this automatically. This prevents aliasing and ensures that frequency-domain features (e.g., CPC) are accurate.

    ```
    rri_ds = mne.filter.resample(rri_filled, up=target_sfreq, down=sfreq_ecg)
    edr_ds = mne.filter.resample(edr_filled, up=target_sfreq, down=sfreq_ecg)
    ```

---

### Extracted CPC Features

```
feat_names_cpc = [
    'stable%', 'unstable%', 'R+W%',
    'log_pwr_HFC', 'log_pwr_LFC', 'log_pwr_VLFC',
    'log_pwr_H2L', 'log_pwr_VL2LH'
]
```

These describe how much the heart-breathing coupling is stable or unstable.

---

## 🌬️ HRnadir – Hear Rate Nadir
### What is it:
The HR nadir represents the lowest heart rate observed during sleep, along with the time and sleep stage at which it occurs. It is a useful marker for autonomic activity and cardiovascular dynamics during sleep.

### Steps to compute HR Nadir:
1.	Compute HR trace from artifact-free R-peaks using NeuroKit2.
2.	Remove outliers: set HR values below 20 bpm to NaN.
3.	Smooth HR using a 10-second rolling median to reduce noise.
4.	Mask missing sleep stages: ignore HR values corresponding to undefined sleep stage epochs.
5.	Find minimum HR: identify the HR nadir in the processed signal.
6.	Map sleep stage at the nadir: convert numeric stage codes to text labels (W, N1, N2, N3, REM).

This procedure ensures that the computed HR nadir is robust, physiologically meaningful, and aligned with sleep stages.

---

Last updated: October 2025
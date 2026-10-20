# Conductivity Fluctuation Analysis of Thin-Film Materials

Analysis of 1/f conductivity noise in thin-film Bi, Cr, and Fe samples using
binary `.Q00` measurement files.

The project extracts spectral features from conductivity fluctuation signals,
trains material-recognition models, and analyzes possible changes in noise
characteristics after electromagnetic pulse irradiation.

---

## Project Goal

The main objective is to determine whether conductivity fluctuation noise can
be used as a non-destructive indicator of thin-film material type and
radiation-induced structural changes.

The project addresses two related tasks:

1. **Material recognition** using spectral noise features.
2. **Radiation-effect analysis** by comparing stable and irradiated films.

---

## Physical Background

Conductivity fluctuations in thin metallic films often exhibit a `1/f` noise
component. In a log-log power spectral density plot, this appears as a linear
dependence:

\[
\log S(f) = \gamma \log f + C
\]

where:

- \(S(f)\) is the power spectral density;
- \(f\) is frequency;
- \(\gamma\) is the spectral exponent;
- \(C\) is a constant.

The spectral exponent \(\gamma\) characterizes the slope of the noise spectrum.
The average spectral energy describes the overall intensity of conductivity
fluctuations.

These two features are used as the primary inputs for clustering and
classification.

---

## Dataset

The source data are binary `.Q00` files containing conductivity fluctuation
measurements.

Each file contains:

- calibration coefficients;
- measurement metadata;
- raw ADC signal values;
- time-related record information.

The dataset includes three material groups and one reference group:

| Group | Description | Role |
|---|---|---|
| `BI` | Bismuth thin films | Target material |
| `CR` | Chromium thin films | Target material |
| `FE` | Iron thin films | Target material |
| `XXX` | Modal reference film | Composition-independent reference |

### Stable training films

The following stable films are used for supervised material recognition:

| File | Material | Year | Measurement period |
|---|---|---:|---|
| `BI202C01` | Bi | 2005 | 2005-08-14 – 2005-08-29 |
| `BI304C01` | Bi | 2005 | 2005-09-02 – 2005-10-13 |
| `CR202C01` | Cr | 2005 | 2005-08-14 – 2005-08-29 |
| `CR304C01` | Cr | 2005 | 2005-09-02 – 2005-10-13 |
| `FE202C01` | Fe | 2005 | 2005-08-14 – 2005-08-29 |
| `FE104C01` | Fe | 2005 | 2005-09-02 – 2005-10-13 |

### Irradiated films

Irradiated films are **not used for model training**. They are analyzed
separately to evaluate possible radiation-induced changes.

| File | Material | Year | Measurement period |
|---|---|---:|---|
| `BI105C01` | Bi | 2006 | 2006-03-31 – 2006-04-29 |
| `BI106C01` | Bi | 2006 | 2006-04-29 – 2006-06-30 |
| `CR205C01` | Cr | 2006 | 2006-03-31 – 2006-04-29 |
| `CR206C01` | Cr | 2006 | 2006-04-29 – 2006-06-30 |
| `FE105C01` | Fe | 2006 | 2006-03-31 – 2006-04-29 |
| `FE106C01` | Fe | 2006 | 2006-04-29 – 2006-06-30 |

### Reference films

`XXX01C01`, `XXX01C03`, and `XXX01C05` are modal reference films. They are
excluded from supervised material classification because they do not represent
the Bi, Cr, or Fe material classes.

The channel suffixes `C01`, `C03`, and `C05` identify measurement channels or
records. Their physical meaning must be taken from the experimental
documentation; it cannot be inferred reliably from the channel number alone.

---

## Project Structure

```text
conductivity-fluctuation-analysis/
├── data/
│   ├── q00/
│   │   ├── BI202C01.Q00
│   │   ├── BI304C01.Q00
│   │   ├── ...
│   │   └── XXX01C05.Q00
│   └── processed/
│       ├── BI202C01_features.csv
│       ├── ...
│       └── conductivity_features.csv
├── notebooks/
│   ├── 01_signal_processing.ipynb
│   └── 02_machine_learning.ipynb
├── src/
│   ├── signal_processing.py
│   ├── train_models.py
│   └── train_knn_raw_samples.py
├── results/
│   ├── figures/
│   └── metrics/
├── requirements.txt
└── README.md
```

---

## Installation

### 1. Clone the repository

```bash
git clone [https://github.com/Katterina07/conductivity-fluctuation-analysis.git](https://github.com/Katterina07/conductivity-fluctuation-analysis.git)
cd conductivity-fluctuation-analysis
```

### 2. Create a virtual environment

On Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Required Data

Place the original binary measurement files in:

```text
data/q00/
```

The expected file names are:

```text
BI202C01.Q00
BI304C01.Q00
CR202C01.Q00
CR304C01.Q00
FE202C01.Q00
FE104C01.Q00
BI105C01.Q00
BI106C01.Q00
CR205C01.Q00
CR206C01.Q00
FE105C01.Q00
FE106C01.Q00
XXX01C01.Q00
XXX01C03.Q00
XXX01C05.Q00
```

The signal-processing script processes exactly these 15 files. If an expected
file is missing, it is skipped with a warning. Additional `.Q00` files present
in `data/q00/` but not listed in `ALL_FILES` in `src/signal_processing.py`
are not processed.

---

## Signal Processing Pipeline

The pipeline is implemented in `src/signal_processing.py`.

For each `.Q00` file, the script:

1. Reads binary records, including calibration and metadata fields.
2. Splits the signal into overlapping samples.
3. Removes a linear trend from each sample.
4. Applies a Blackman window.
5. Computes the one-sided spectrum.
6. Estimates the spectral exponent \(\gamma\).
7. Computes the average spectral energy.
8. Saves per-file and combined feature datasets.

The following overlap modes are supported:

| Mode | Overlap |
|---:|---:|
| 0 | No overlap |
| 1 | 25% |
| 2 | 50% |
| 3 | 75% |

The default configuration uses:

- sample length: `128`;
- overlap mode: `2` (50% overlap);
- spectral features: `gamma` and `log_energy`.

---

## Feature Description

The combined dataset contains the following main columns:

| Column | Description |
|---|---|
| `gamma` | Spectral exponent from log-log power spectrum fitting |
| `energy` | Average spectral power |
| `log_energy` | Base-10 logarithm of spectral energy |
| `file` | Source `.Q00` file name |
| `material` | Material label: `BI`, `CR`, `FE`, or `XXX` |
| `material_name` | Human-readable material name |
| `year` | Measurement year |
| `measurement_period` | Measurement date range |
| `irradiated` | Whether the film was irradiated |
| `sampling_rate_hz` | Sampling rate used for spectral analysis |
| `used_for_training` | Whether the sample belongs to a stable training film |

---

## Models

The analysis uses three approaches:

### K-Means Clustering

K-Means is used to check whether stable Bi, Cr, and Fe films form separable
groups in the two-dimensional feature space:

- spectral exponent \(\gamma\);
- logarithmic spectral energy.

The clustering quality is estimated using the silhouette score.

### K-Nearest Neighbors

A KNN classifier is trained on stable films only. It provides a simple
non-parametric baseline for material recognition.

### Fully Connected Neural Network

A multilayer perceptron is used for supervised material classification.

The neural network is trained using:

- standardized `gamma` and `log_energy`;
- a stratified train-test split;
- early stopping;
- a fixed random seed for reproducibility.

---

## Radiation-Effect Analysis

Irradiated films are excluded from training. After the neural network is
trained on stable films, it is applied to irradiated films to estimate whether
their spectral features remain close to the original material class.

The analysis includes:

- predicted material class;
- prediction confidence;
- comparison of stable and irradiated `gamma`;
- comparison of stable and irradiated `log_energy`;
- visual comparison of feature distributions.

---

## How to Run

### 1. Process `.Q00` files

```bash
python src/signal_processing.py
```

This creates:

```text
data/processed/BI202C01_features.csv
data/processed/BI304C01_features.csv
...
data/processed/XXX01C05_features.csv
data/processed/conductivity_features.csv
```

### 2. Train models and analyze irradiated films

```bash
python src/train_models.py
```

This creates:

```text
results/figures/kmeans_clusters.png
results/figures/neural_network_confusion_matrix.png
results/figures/file_level_gamma_comparison.png
results/figures/file_level_energy_comparison.png
results/figures/gamma_boxplots_by_file.png
results/figures/energy_boxplots_by_file.png

results/metrics/model_metrics.csv
results/metrics/irradiated_film_predictions.csv
```

### 3. Optional: KNN on raw `.Q03` samples

```bash
python src/train_knn_raw_samples.py
```

This script is separate from the `.Q00` spectral-feature pipeline. It reads
fixed-length float32 samples from `data/q03/*.Q03`, trains a KNN classifier,
and creates:

```text
results/figures/knn_raw_samples_confusion_matrix.png
results/figures/class_distribution.png

results/metrics/knn_raw_samples_metrics.csv
```

This script requires `.Q03` files in `data/q03/`. It is not used to produce
`conductivity_features.csv` or the spectral-feature models.

---

## Notebooks

### `notebooks/01_signal_processing.ipynb`

Demonstrates and checks the `.Q00` processing pipeline implemented in
`src/signal_processing.py`:

- reading binary Q00 records;
- splitting signals into overlapping samples;
- detrending and windowing;
- computing `gamma`, `energy`, and `log_energy`;
- creating per-file and combined CSV datasets;
- visualizing feature distributions.

### `notebooks/02_machine_learning.ipynb`

Reproduces the machine-learning pipeline implemented in
`src/train_models.py`:

- loading `data/processed/conductivity_features.csv`;
- selecting stable training films;
- creating a stratified train-test split;
- standardizing features;
- training K-Means, KNN, and an MLP classifier;
- evaluating classification quality;
- applying the trained MLP to irradiated films;
- saving metrics and prediction files.

Run `src/signal_processing.py` before running the ML notebook.

---

## Results

The main result figures are:

| Figure | Purpose |
|---|---|
| `kmeans_clusters.png` | Shows clustering of stable film noise features |
| `neural_network_confusion_matrix.png` | Shows material-recognition performance |
| `file_level_gamma_comparison.png` | Compares file-level spectral exponent for stable and irradiated films |
| `file_level_energy_comparison.png` | Compares file-level spectral energy for stable and irradiated films |
| `gamma_boxplots_by_file.png` | Shows the distribution of \(\gamma\) by measurement file |
| `energy_boxplots_by_file.png` | Shows the distribution of log spectral energy by measurement file |

The main metrics are saved in:

```text
results/metrics/model_metrics.csv
```

Predictions for irradiated films are saved in:

```text
results/metrics/irradiated_film_predictions.csv
```

---

## Important Note on Sampling Rate

The spectral analysis uses the sampling rate specified in `FILE_METADATA`
inside `src/signal_processing.py`.

For the main material measurements, the current value is:

```python
"sampling_rate_hz": 1.73
```

For the reference films, the current value is:

```python
"sampling_rate_hz": 1.0
```

If the raw `.Q00` signal is sampled at a different internal rate, update the
`sampling_rate_hz` field for the corresponding files. The sampling rate must
correspond to the time interval between individual signal points used in the
spectral analysis.

---

## Reproducibility

The following parameters are fixed:

- random seed: `42`;
- sample length: `128`;
- overlap mode: `2`;
- train-test split: `75% / 25%`;
- stratified sampling by material class.

This allows the preprocessing, clustering, classification, and radiation-effect
analysis to be reproduced consistently.

---

## Methodological Limitations

- The current train-test split is performed at the level of overlapping
  spectral samples. Because neighboring windows overlap by 50%, test samples
  may be highly similar to training samples. For a stricter estimate,
  use file-level or record-level splitting.
- The training set contains only six stable files, with two files per material.
  Therefore, classification results should be interpreted cautiously.
- `energy` is the mean of windowed spectral power and is not normalized as a
  physical power spectral density.
- The irradiation status, measurement periods, material labels, and channel
  meanings must be confirmed from the experimental documentation.

---

## Technologies

- Python 3
- NumPy
- pandas
- SciPy
- scikit-learn
- Matplotlib
- Seaborn
- Jupyter

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
component. In a log-log power spectral density plot, this appears as a
linear dependence:

\[
\log S(f) = \gamma \log f + C
\]

where:

- \( S(f) \) is the power spectral density;
- \( f \) is frequency;
- \( \gamma \) is the spectral exponent;
- \( C \) is a constant.

The spectral exponent \( \gamma \) characterizes the slope of the noise
spectrum. The average spectral energy describes the overall intensity of
conductivity fluctuations.

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

---

## Signal Processing Pipeline

The pipeline is implemented in `src/signal_processing.py`.

For each `.Q00` file, the script:

1. Reads binary records, including calibration and metadata fields.
2. Splits the signal into overlapping samples.
3. Removes a linear trend from each sample.
4. Applies a Blackman window.
5. Computes the one-sided amplitude spectrum.
6. Estimates the spectral exponent \( \gamma \).
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

- spectral exponent \( \gamma \);
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

This makes it possible to identify whether electromagnetic pulse irradiation
produces measurable changes in conductivity fluctuation noise.

---

## Project Structure

```text
conductivity-noise-analysis/
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
├── src/
│   ├── signal_processing.py
│   └── train_models.py
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
git clone [https://github.com/Katterina07/conductivity-noise-analysis.git](https://github.com/Katterina07/conductivity-noise-analysis.git)
cd conductivity-noise-analysis
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
results/figures/stable_vs_irradiated_films.png
results/figures/irradiated_gamma_comparison.png
results/figures/irradiated_energy_comparison.png

results/metrics/model_metrics.csv
results/metrics/irradiated_film_predictions.csv
```

---

## Results

The main result figures are:

| Figure | Purpose |
|---|---|
| `kmeans_clusters.png` | Shows clustering of stable film noise features |
| `neural_network_confusion_matrix.png` | Shows material-recognition performance |
| `stable_vs_irradiated_films.png` | Compares stable and irradiated films |
| `irradiated_gamma_comparison.png` | Shows changes in spectral exponent |
| `irradiated_energy_comparison.png` | Shows changes in spectral energy |

The main metrics are saved in:

```text
results/metrics/model_metrics.csv
```

---

## Important Note on Sampling Rate

The spectral analysis uses the sampling rate specified in
`FILE_METADATA` inside `src/signal_processing.py`.

For the main material measurements, the current value is:

```python
"sampling_rate_hz": 1.73
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

## Technologies

- Python 3
- NumPy
- pandas
- SciPy
- scikit-learn
- Matplotlib
- Seaborn

---

## Possible Extensions

- Add Welch averaging for more stable spectral estimates.
- Use multitaper spectral estimation.
- Add wavelet-based features.
- Analyze individual frequency bands separately.
- Add statistical hypothesis testing for stable vs irradiated films.
- Extend the model to multi-file and multi-session validation.
- Add uncertainty estimates for predicted material classes.

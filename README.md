# Conductivity Fluctuation Analysis

This repository contains a reproducible Python pipeline for processing low-frequency
conductivity fluctuation measurements of thin metallic films and evaluating
machine-learning models for material recognition.

The workflow processes binary `.Q00` measurement files, extracts spectral features,
and evaluates K-Means clustering, K-Nearest Neighbors (KNN), and a fully connected
neural network.

## Results

The following results were obtained using spectral features computed from the
conductivity fluctuation signals:

| Model | Metric | Reproduced result | Result reported in the thesis |
|---|---:|---:|---:|
| K-Means | Silhouette score | 0.5098 | Qualitative cluster separation |
| K-Nearest Neighbors | Accuracy | **86.10%** | 86% |
| Fully connected neural network | Accuracy | **88.23%** | 89.41% |

The KNN result closely reproduces the result reported in the thesis. The neural-network
result differs by 1.18 percentage points, which is consistent with variation caused by
random weight initialization, training order, and software-version differences.

The main result of the original thesis — **89.41% accuracy** — was obtained with a
fully connected neural network using spectral features, rather than with KNN on raw
time-domain samples.

## Data and features

The input data are binary `.Q00` files containing conductivity fluctuation
measurements for bismuth (Bi), chromium (Cr), and iron (Fe) thin films.
Reference measurements (`XXX`) are used to identify composition-independent
contributions from the measurement system.

For each 128-point sample, the pipeline computes:

- mean fluctuation energy;
- spectral shape exponent gamma;
- additional file-level spectral characteristics used in the analysis notebooks.

The spectral features are used for supervised material classification.

## Machine-learning models

- **K-Means clustering**: unsupervised analysis with `k = 4`.
- **KNN**: supervised classification using spectral features.
- **Fully connected neural network**: supervised classification using spectral features.

The neural network uses four hidden layers with 128 neurons and was trained for
100 epochs, following the methodology described in the thesis.

## Repository structure

```text
data/
  q00/                  Raw binary measurement files (not distributed)
  processed/            Extracted spectral features
notebooks/
  01_signal_processing.ipynb
  02_machine_learning.ipynb
results/
  figures/              Generated plots
  metrics/              Model metrics and prediction outputs
src/
  signal_processing.py  Q00 reading, calibration, sample extraction, features
  train_models.py       Spectral-feature model training and evaluation
```

## Requirements

- Python 3.10 or newer
- NumPy
- pandas
- SciPy
- scikit-learn
- Matplotlib
- Seaborn
- Jupyter / nbconvert

Install the dependencies:

```bash
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

On macOS or Linux, use:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## How to reproduce

Run the signal-processing pipeline:

```bash
python src/signal_processing.py
```

Then execute and save the notebooks:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/01_signal_processing.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/02_machine_learning.ipynb
```

The notebooks generate figures, model metrics, and prediction outputs in the
`results/` directory.

## Data availability

Raw `.Q00` measurement files are not distributed in this repository.

Processed feature tables, model metrics, figures, and analysis notebooks are
included to make the computational workflow and reported results reproducible.

## Notes on evaluation

The reported KNN and neural-network accuracies use a random sample-level
train-test split, following the evaluation protocol of the original thesis.

Because the samples contain overlapping windows from the same measurement file,
this protocol can overestimate generalization to entirely unseen files. A
file-level leave-one-file-out evaluation of KNN using raw 128-point samples is
also included as a diagnostic result and shows substantially lower accuracy.

Therefore, the primary reproduced results are based on spectral features, while
raw-sample classification is treated as an additional exploratory analysis.

## Limitations

- The number of independent measurement files is limited: there are two files
  for each of the three studied materials.
- Overlapping samples from the same `.Q00` file are statistically dependent.
- A random sample-level train-test split may therefore overestimate performance
  on completely unseen measurement files.
- The reported neural-network result may vary between runs because of random
  weight initialization and training stochasticity.
- Raw 128-point conductivity samples do not generalize well across measurement
  files; spectral features are more suitable for the material-classification task.

## Citation

If you use this repository or its methodology, please cite the underlying
bachelor’s thesis:

> Chekalina, E. A. *Fluctuations of the Conductivity of Thin Metal Films*.
> South Ural State University, 2024.
# Conductivity Fluctuation Analysis in Thin Metal Films

Machine learning project based on my Bachelor’s thesis in Applied Mathematics and Physics.

The project analyzes low-frequency conductivity fluctuations in bismuth, chromium, and iron thin films. It combines experimental time-series processing, spectral feature extraction, clustering, and classification.

## Project Goal

To determine whether conductivity fluctuation patterns can be used to distinguish thin metal films of different compositions and to separate composition-independent noise using a modal reference film.

## Key Result

The thesis pipeline achieved **89.41% recognition accuracy** in distinguishing conductivity fluctuation patterns across films of different compositions.

Methods used in the thesis:

- K-Means clustering
- K-Nearest Neighbors (KNN)
- Fully connected neural network

## Data

The project uses experimental conductivity fluctuation measurements from thin metal films.

The original experimental dataset cannot be shared publicly. The repository expects CSV or binary input files with time-series conductivity measurements.

Material groups:

- BI — bismuth films
- CR — chromium films
- FE — iron films

Reference group:

- XXX — modal reference film

The reference film was used to identify conductivity fluctuation components that are not related to the chemical composition of the material films.

## Methods

### 1. Signal processing

- Reading binary experimental data
- Splitting long signals into overlapping samples
- Removing linear trends
- Applying a Blackman window
- Computing FFT-based power spectra
- Extracting 1/f noise spectrum parameters
- Calculating average spectral energy

### 2. Machine learning

Two modeling approaches are implemented:

| Approach | Input | Features | Models |
|---|---|---|---|
| Raw sample classification | Fixed-length time-series samples | Raw conductivity fluctuation signal | KNN |
| Feature-based analysis | Processed time-series samples | Gamma parameter, spectral energy | K-Means, KNN, neural network |

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── data/
│   └── README.md
├── notebooks/
│   ├── 01_signal_processing.ipynb
│   └── 02_machine_learning.ipynb
├── src/
│   ├── signal_processing.py
│   ├── build_dataset.py
│   ├── train_models.py
│   └── train_knn_raw_samples.py
├── results/
│   ├── figures/
│   └── metrics/
└── docs/
    └── thesis_summary.md
```

## Installation

```bash
git clone [https://github.com/Katterina07/conductivity-fluctuation-analysis.git](https://github.com/Katterina07/conductivity-fluctuation-analysis.git)
cd conductivity-fluctuation-analysis

python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

pip install -r requirements.txt
```

## Usage

### 1. Process raw experimental data

```bash
python src/build_dataset.py
```

### 2. Train and evaluate feature-based models

```bash
python src/train_models.py
```

### 3. Classify raw samples with KNN

```bash
python src/train_knn_raw_samples.py
```

## Results

After running the pipeline, the following outputs are created:

```text
results/
├── figures/
│   ├── feature_distribution.png
│   ├── kmeans_clusters.png
│   ├── neural_network_confusion_matrix.png
│   ├── knn_raw_samples_confusion_matrix.png
│   ├── reference_vs_materials_gamma.png
│   └── reference_vs_materials_energy.png
└── metrics/
    ├── model_metrics.csv
    ├── knn_raw_samples_metrics.csv
    ├── reference_film_summary.csv
    └── material_film_summary.csv
```

## Important Note

This repository contains a **reconstructed and cleaned implementation** of the data-processing and machine-learning pipeline described in my Bachelor’s thesis.

The original final code was not fully preserved. This version was rebuilt from earlier working notebooks and the methodology described in the thesis.

## Tools

- Python
- NumPy
- Pandas
- SciPy
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter / Google Colab

## License

This project is released under the MIT License.

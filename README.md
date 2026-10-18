# conductivity-fluctuation-analysis
Machine learning analysis of conductivity fluctuations in thin metal films: Python, time-series processing, K-Means, KNN, and neural networks.
# Conductivity Fluctuation Analysis of Thin Metal Films

Analysis of experimental conductivity fluctuation data for thin iron, bismuth, 
and chromium films using Python and machine learning methods.

## Project Overview

This project investigates whether characteristics of conductivity fluctuations 
can be used to identify patterns associated with the chemical composition of 
thin metal films.

The project includes data preprocessing, exploratory data analysis, feature 
extraction from time series, visualization, and the application of machine 
learning models.

## Objectives

- Process experimental conductivity data for Fe, Bi, and Cr thin films.
- Develop a Python-based pipeline for preliminary time-series processing.
- Calculate statistical characteristics of conductivity fluctuations.
- Perform exploratory data analysis and identify outliers and patterns.
- Apply machine learning methods to assess relationships between signal 
  characteristics and film composition.
- Compare ML approaches and evaluate their suitability for the task.

## Data

- **Source:** Experimental data from the works of A. A. Shulginov and 
  N. S. Zabeyvorota.
- **Objects:** Thin metal films of iron (Fe), bismuth (Bi), and chromium (Cr).
- **Data type:** Time series of conductivity measurements.
- **Note:** The raw experimental data are not publicly available in this 
  repository.

## Methods and Tools

- **Programming language:** Python
- **Data processing:** pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine learning:** scikit-learn
- **Development environment:** Jupyter Notebook
- **Version control:** Git, GitHub

## Project Structure

```text
conductivity-films-analysis/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/                  # Original experimental data
│   └── processed/            # Cleaned and transformed data
├── notebooks/
│   ├── 01_eda.ipynb          # Exploratory data analysis
│   ├── 02_preprocessing.ipynb
│   └── 03_ml_models.ipynb
├── src/
│   ├── preprocessing.py      # Data preprocessing functions
│   ├── features.py           # Feature extraction
│   └── models.py             # Model training and evaluation
├── reports/
│   └── figures/              # Generated plots and visualizations
└── LICENSE
```

## Key Results

- Developed a reproducible Python pipeline for processing conductivity 
  fluctuation data.
- Performed exploratory analysis and visualization of time-series signals.
- Extracted features characterizing conductivity fluctuations.
- Applied and compared machine learning methods for analyzing relationships 
  between signal characteristics and film composition.
- Identified suitable ML approaches for experimental data analysis.

## How to Run

1. Clone the repository:

```bash
git clone [https://github.com/username/conductivity-films-analysis.git](https://github.com/username/conductivity-films-analysis.git)
cd conductivity-films-analysis
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Open Jupyter Notebook:

```bash
jupyter notebook
```

## Notes

The original experimental dataset is not included in this repository. 
The repository demonstrates the full analytical workflow: data preprocessing, 
exploratory analysis, feature engineering, visualization, and machine learning.

## License

This project is licensed under the MIT License.

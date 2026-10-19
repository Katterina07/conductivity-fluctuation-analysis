"""
Build a feature dataset from conductivity fluctuation .Q00 measurements.

The pipeline extracts 1/f noise features and saves material metadata,
measurement conditions, and radiation-exposure information.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import welch


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "q00"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
FIGURES_DIR = BASE_DIR / "results" / "figures"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

DATASET_PATH = PROCESSED_DIR / "conductivity_features.csv"

REFERENCE_LABEL = "XXX"
MATERIAL_ORDER = ["BI", "CR", "FE"]

MATERIAL_NAMES = {
    "BI": "Bismuth (Bi)",
    "CR": "Chromium (Cr)",
    "FE": "Iron (Fe)",
    "XXX": "Reference film",
}

MATERIAL_COLORS = {
    "BI": "#4C72B0",
    "CR": "#DD8452",
    "FE": "#55A868",
    "XXX": "#000000",
}

# Training files: stable non-irradiated films
TRAINING_FILES = {
    "bi202c01": "BI",
    "bi304c01": "BI",
    "cr202c01": "CR",
    "cr304c01": "CR",
    "fe202c01": "FE",
    "fe104c01": "FE",
}

# Reference films: composition-independent noise
REFERENCE_FILES = {
    "XXX01C01": "XXX",
    "XXX01C03": "XXX",
    "XXX01C05": "XXX",
}

# Irradiated films: used for scientific analysis, not supervised training
IRRADIATED_FILES = {
    "bi105c01": "BI",
    "bi106c01": "BI",
    "cr205c01": "CR",
    "cr206c01": "CR",
    "fe105c01": "FE",
    "fe106c01": "FE",
}

FILE_METADATA = {
    "bi202c01": {
        "material": "BI",
        "year": 2005,
        "irradiated": False,
        "measurement_period": "2005-08-14_2005-08-29",
        "sampling_rate_hz": 1.73,
    },
    "bi304c01": {
        "material": "BI",
        "year": 2005,
        "irradiated": False,
        "measurement_period": "2005-09-02_2005-10-13",
        "sampling_rate_hz": 1.73,
    },
    "cr202c01": {
        "material": "CR",
        "year": 2005,
        "irradiated": False,
        "measurement_period": "2005-08-14_2005-08-29",
        "sampling_rate_hz": 1.73,
    },
    "cr304c01": {
        "material": "CR",
        "year": 2005,
        "irradiated": False,
        "measurement_period": "2005-09-02_2005-10-13",
        "sampling_rate_hz": 1.73,
    },
    "fe202c01": {
        "material": "FE",
        "year": 2005,
        "irradiated": False,
        "measurement_period": "2005-08-14_2005-08-29",
        "sampling_rate_hz": 1.73,
    },
    "fe104c01": {
        "material": "FE",
        "year": 2005,
        "irradiated": False,
        "measurement_period": "2005-09-02_2005-10-13",
        "sampling_rate_hz": 1.73,
    },
    "bi105c01": {
        "material": "BI",
        "year": 2006,
        "irradiated": True,
        "measurement_period": "2006-03-31_2006-04-29",
        "sampling_rate_hz": 1.73,
    },
    "bi106c01": {
        "material": "BI",
        "year": 2006,
        "irradiated": True,
        "measurement_period": "2006-04-29_2006-06-30",
        "sampling_rate_hz": 1.73,
    },
    "cr205c01": {
        "material": "CR",
        "year": 2006,
        "irradiated": True,
        "measurement_period": "2006-03-31_2006-04-29",
        "sampling_rate_hz": 1.73,
    },
    "cr206c01": {
        "material": "CR",
        "year": 2006,
        "irradiated": True,
        "measurement_period": "2006-04-29_2006-06-30",
        "sampling_rate_hz": 1.73,
    },
    "fe105c01": {
        "material": "FE",
        "year": 2006,
        "irradiated": True,
        "measurement_period": "2006-03-31_2006-04-29",
        "sampling_rate_hz": 1.73,
    },
    "fe106c01": {
        "material": "FE",
        "year": 2006,
        "irradiated": True,
        "measurement_period": "2006-04-29_2006-06-30",
        "sampling_rate_hz": 1.73,
    },
    "XXX01C01": {
        "material": "XXX",
        "year": None,
        "irradiated": False,
        "measurement_period": "reference",
        "sampling_rate_hz": 1.0,
    },
    "XXX01C03": {
        "material": "XXX",
        "year": None,
        "irradiated": False,
        "measurement_period": "reference",
        "sampling_rate_hz": 1.0,
    },
    "XXX01C05": {
        "material": "XXX",
        "year": None,
        "irradiated": False,
        "measurement_period": "reference",
        "sampling_rate_hz": 1.0,
    },
}

ALL_FILES = {
    **TRAINING_FILES,
    **REFERENCE_FILES,
    **IRRADIATED_FILES,
}


def material_name(label):
    return MATERIAL_NAMES.get(label, label)


def load_q00_signal(file_path, sampling_rate):
    signal = np.fromfile(file_path, dtype="<f4")
    time = np.arange(len(signal)) / sampling_rate
    return signal, time


def extract_spectrum_features(signal, sampling_rate):
    frequencies, psd = welch(
        signal,
        fs=sampling_rate,
        nperseg=min(4096, len(signal)),
        detrend="constant",
    )

    mask = frequencies > 0
    frequencies = frequencies[mask]
    psd = psd[mask]

    log_frequencies = np.log10(frequencies)
    log_psd = np.log10(psd.clip(lower=1e-30))

    slope, intercept = np.polyfit(log_frequencies, log_psd, 1)

    return {
        "gamma": -slope,
        "energy": np.trapz(psd, frequencies),
        "frequencies": frequencies,
        "psd": psd,
    }


def build_dataset():
    rows = []
    spectra = {}

    for file_stem, material in ALL_FILES.items():
        file_path = DATA_DIR / f"{file_stem}.Q00"

        if not file_path.exists():
            print(f"Skipping missing file: {file_path}")
            continue

        metadata = FILE_METADATA[file_stem]
        sampling_rate = metadata["sampling_rate_hz"]

        signal, time = load_q00_signal(file_path, sampling_rate)
        features = extract_spectrum_features(signal, sampling_rate)

        rows.append(
            {
                "file": file_stem,
                "material": material,
                "material_name": material_name(material),
                "year": metadata["year"],
                "measurement_period": metadata["measurement_period"],
                "irradiated": metadata["irradiated"],
                "sampling_rate_hz": sampling_rate,
                "gamma": features["gamma"],
                "energy": features["energy"],
                "log_energy": np.log10(max(features["energy"], 1e-30)),
                "n_samples": len(signal),
                "used_for_training": file_stem in TRAINING_FILES,
            }
        )

        spectra[file_stem] = features

        print(
            f"{file_stem}: {material_name(material)}, "
            f"gamma={features['gamma']:.3f}, "
            f"energy={features['energy']:.3e}"
        )

    if not rows:
        raise RuntimeError("No .Q00 files were processed.")

    return pd.DataFrame(rows), spectra


def plot_psd(spectra):
    plt.figure(figsize=(12, 7))

    for file_stem, features in spectra.items():
        material = ALL_FILES[file_stem]

        plt.loglog(
            features["frequencies"],
            features["psd"],
            label=f"{material_name(material)} — {file_stem}",
            color=MATERIAL_COLORS[material],
            alpha=0.8,
            linewidth=1.1,
        )

    plt.xlabel("Frequency, Hz", fontsize=12)
    plt.ylabel("Power spectral density", fontsize=12)
    plt.title(
        "Power Spectral Density of Conductivity Fluctuations",
        fontsize=14,
        pad=15,
    )
    plt.legend(title="Material / file", fontsize=8, title_fontsize=10, ncol=2)
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "psd_all_files.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_feature_distributions(df):
    for column, xlabel, filename in [
        ("gamma", "Spectral exponent γ", "gamma_distribution.png"),
        ("log_energy", "Log10 spectral energy", "energy_distribution.png"),
    ]:
        plt.figure(figsize=(10, 6))

        for material in MATERIAL_ORDER:
            subset = df[df["material"] == material]

            plt.hist(
                subset[column],
                bins=20,
                alpha=0.6,
                label=material_name(material),
                color=MATERIAL_COLORS[material],
                edgecolor="black",
                linewidth=0.4,
            )

        reference = df[df["material"] == REFERENCE_LABEL]

        if not reference.empty:
            plt.hist(
                reference[column],
                bins=20,
                histtype="step",
                linewidth=2.5,
                linestyle="--",
                color="black",
                label=material_name(REFERENCE_LABEL),
            )

        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel("Number of files", fontsize=12)
        plt.title(
            f"Distribution of {xlabel}",
            fontsize=14,
            pad=12,
        )
        plt.legend(title="Material")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / filename, dpi=300, bbox_inches="tight")
        plt.close()


def main():
    df, spectra = build_dataset()

    df.to_csv(DATASET_PATH, index=False)

    plot_psd(spectra)
    plot_feature_distributions(df)

    print(f"Dataset saved: {DATASET_PATH}")
    print(f"Figures saved: {FIGURES_DIR}")
    print(f"Processed files: {len(df)}")


if __name__ == "__main__":
    main()

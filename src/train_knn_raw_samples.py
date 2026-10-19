"""
KNN classification using raw conductivity fluctuation samples.

Each .Q03 file contains fixed-length float32 samples.

XXX is a modal reference film used to identify composition-independent
noise. It is excluded from supervised material classification.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "q03"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"

REFERENCE_LABEL = "XXX"
MATERIAL_ORDER = ["BI", "CR", "FE"]

SAMPLE_LENGTH = 128
TEST_SIZE = 0.2
RANDOM_STATE = 42

FILE_LABELS = {
    "BI202C01": "BI",
    "BI304C01": "BI",
    "CR202C01": "CR",
    "CR304C01": "CR",
    "FE104C01": "FE",
    "FE202C01": "FE",
    "XXX01C01": "XXX",
    "XXX01C03": "XXX",
    "XXX01C05": "XXX",
}

MATERIAL_NAMES = {
    "BI": "Bismuth (Bi)",
    "CR": "Chromium (Cr)",
    "FE": "Iron (Fe)",
    "XXX": "Reference film",
}


def get_material_name(label: str) -> str:
    """
    Convert a short material code into a readable plot label.
    """
    return MATERIAL_NAMES.get(label, label)


def load_q03_samples(file_path: Path, sample_length: int):
    """
    Read fixed-length float32 samples from a binary .Q03 file.
    """
    record_size = sample_length * 4

    data = file_path.read_bytes()

    if len(data) % record_size != 0:
        print(
            f"Warning: {file_path.name} size is not divisible by "
            f"{record_size} bytes. Truncating incomplete final sample."
        )

        usable_bytes = (len(data) // record_size) * record_size
        data = data[:usable_bytes]

    samples = np.frombuffer(data, dtype="<f4")

    return samples.reshape(-1, sample_length)


def build_dataset():
    """
    Load .Q03 files and build X, y arrays for material classification.
    """
    X_parts = []
    y_parts = []

    for file_stem, material in FILE_LABELS.items():
        if material == REFERENCE_LABEL:
            print(
                f"Skipping reference film {file_stem}: "
                "not used for supervised material classification."
            )
            continue

        file_path = DATA_DIR / f"{file_stem}.Q03"

        if not file_path.exists():
            print(f"Skipping missing file: {file_path}")
            continue

        samples = load_q03_samples(
            file_path=file_path,
            sample_length=SAMPLE_LENGTH,
        )

        labels = np.full(len(samples), material)

        X_parts.append(samples)
        y_parts.append(labels)

        print(
            f"{file_stem}: {len(samples)} samples, "
            f"label={get_material_name(material)}"
        )

    if not X_parts:
        raise RuntimeError("No material .Q03 files were found.")

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)

    return X, y


def plot_confusion_matrix(y_test, y_pred, labels, path):
    """
    Save a confusion matrix for the raw-sample KNN model.
    """
    readable_labels = [get_material_name(label) for label in labels]

    cm = confusion_matrix(y_test, y_pred, labels=labels)

    plt.figure(figsize=(9, 7))

    ax = sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=readable_labels,
        yticklabels=readable_labels,
        cbar_kws={"label": "Number of test samples"},
        linewidths=0.5,
        linecolor="white",
        annot_kws={"fontsize": 12},
    )

    ax.set_xlabel("Predicted material", fontsize=12, labelpad=10)
    ax.set_ylabel("True material", fontsize=12, labelpad=10)
    ax.set_title(
        "KNN Classification of Thin-Film Materials\n"
        "Raw Conductivity Fluctuation Samples",
        fontsize=14,
        pad=15,
    )

    plt.xticks(rotation=30, ha="right", fontsize=11)
    plt.yticks(rotation=0, fontsize=11)

    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_class_distribution(y, path):
    """
    Save a bar chart showing the number of samples per material.
    """
    unique_labels, counts = np.unique(y, return_counts=True)

    readable_labels = [get_material_name(label) for label in unique_labels]

    colors = {
        "Bismuth (Bi)": "#4C72B0",
        "Chromium (Cr)": "#DD8452",
        "Iron (Fe)": "#55A868",
    }

    bar_colors = [colors.get(label, "#8172B3") for label in readable_labels]

    plt.figure(figsize=(9, 6))
    bars = plt.bar(readable_labels, counts, color=bar_colors, edgecolor="black")

    for bar, count in zip(bars, counts):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts) * 0.01,
            str(count),
            ha="center",
            va="bottom",
            fontsize=11,
        )

    plt.xlabel("Material", fontsize=12)
    plt.ylabel("Number of samples", fontsize=12)
    plt.title(
        "Class Distribution of Conductivity Fluctuation Samples",
        fontsize=14,
        pad=15,
    )
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    X, y = build_dataset()

    print(f"Dataset shape: {X.shape}")
    print(
        "Class distribution:",
        dict(zip(*np.unique(y, return_counts=True))),
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(
            n_neighbors=5,
            weights="distance",
        ),
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print(f"KNN accuracy: {accuracy * 100:.2f}%")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=MATERIAL_ORDER,
            target_names=[get_material_name(label) for label in MATERIAL_ORDER],
            zero_division=0,
        )
    )

    cv_scores = cross_val_score(
        model,
        X,
        y,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
    )

    print("5-fold CV accuracy:")
    print(f"Mean: {cv_scores.mean() * 100:.2f}%")
    print(f"Std: {cv_scores.std() * 100:.2f}%")

    plot_confusion_matrix(
        y_test=y_test,
        y_pred=y_pred,
        labels=MATERIAL_ORDER,
        path=FIGURES_DIR / "knn_raw_samples_confusion_matrix.png",
    )

    plot_class_distribution(
        y=y,
        path=FIGURES_DIR / "class_distribution.png",
    )

    results = pd.DataFrame(
        [
            {
                "model": "KNN — raw samples",
                "test_accuracy": accuracy,
                "cv_mean_accuracy": cv_scores.mean(),
                "cv_std": cv_scores.std(),
                "n_neighbors": 5,
                "sample_length": SAMPLE_LENGTH,
            }
        ]
    )

    results.to_csv(
        METRICS_DIR / "knn_raw_samples_metrics.csv",
        index=False,
    )

    print(
        "Metrics saved:",
        METRICS_DIR / "knn_raw_samples_metrics.csv",
    )
    print(
        "Figures saved:",
        FIGURES_DIR,
    )


if __name__ == "__main__":
    main()

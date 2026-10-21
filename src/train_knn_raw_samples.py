"""
KNN classification using raw conductivity fluctuation samples from .Q00 files.

This script reuses the Q00 reader, calibration scaling, sample splitting,
and linear detrending implemented in src/signal_processing.py.

Each 128-point calibrated and detrended sample is used directly as a feature
vector for KNN classification.

XXX reference films are excluded from supervised material classification.
Irradiated films are also excluded from training and are intended for a
separate radiation-effect analysis.

The primary evaluation is file-level 3-fold cross-validation using GroupKFold.
A random sample-level train-test split is retained only as an additional,
less conservative baseline.
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import (
    GroupKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = BASE_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from signal_processing import (
    DATA_DIR,
    FIGURES_DIR,
    TRAINING_FILES,
    REFERENCE_FILES,
    IRRADIATED_FILES,
    read_q00_file,
    split_into_samples,
)

METRICS_DIR = BASE_DIR / "results" / "metrics"


SAMPLE_LENGTH = 128
OVERLAP_MODE = 2

TEST_SIZE = 0.25
RANDOM_STATE = 42
N_SPLITS = 3

REFERENCE_LABEL = "XXX"
MATERIAL_ORDER = ["BI", "CR", "FE"]

MATERIAL_NAMES = {
    "BI": "Bismuth (Bi)",
    "CR": "Chromium (Cr)",
    "FE": "Iron (Fe)",
    "XXX": "Reference film",
}

FILE_LABELS = {
    **TRAINING_FILES,
    **REFERENCE_FILES,
    **IRRADIATED_FILES,
}


def get_material_name(label: str) -> str:
    """Convert a material code into a readable label."""
    return MATERIAL_NAMES.get(label, label)


def extract_raw_samples(
    file_path: Path,
    sample_length: int = SAMPLE_LENGTH,
    overlap_mode: int = OVERLAP_MODE,
) -> np.ndarray:
    """
    Read one .Q00 file and extract calibrated, detrended fixed-length samples.

    The processing matches src/signal_processing.py:
    - read raw ADC codes;
    - calculate the calibration scale;
    - split the signal into overlapping samples;
    - remove a linear trend from each sample.

    Returns:
        np.ndarray: array of shape (n_samples, sample_length).
    """
    records = list(read_q00_file(file_path))

    if not records:
        raise ValueError(f"No valid records found in {file_path.name}")

    all_samples = []

    for record in records:
        scale = 2.0 / (
            record["kc"]
            * record["ku"]
            * record["km"]
            * (record["up1"] + record["up2"])
        )

        samples = split_into_samples(
            codes=record["codes"],
            sample_length=sample_length,
            overlap_mode=overlap_mode,
        )

        time_index = np.arange(sample_length)

        for sample in samples:
            sample = np.asarray(sample, dtype=float)

            scaled_sample = sample * scale
            all_samples.append(scaled_sample)

    if not all_samples:
        return np.empty((0, sample_length), dtype=float)

    return np.asarray(all_samples)


def build_dataset():
    """
    Build X, y, and file-group arrays from stable training .Q00 files.

    XXX reference films and irradiated films are excluded from supervised
    material classification.
    """
    X_parts = []
    y_parts = []
    group_parts = []

    for file_stem, material in FILE_LABELS.items():
        if material == REFERENCE_LABEL:
            print(
                f"Skipping reference film {file_stem}: "
                "not used for supervised material classification."
            )
            continue

        if file_stem not in TRAINING_FILES:
            print(
                f"Skipping irradiated film {file_stem}: "
                "not used for supervised material classification."
            )
            continue

        file_path = DATA_DIR / f"{file_stem}.Q00"

        if not file_path.exists():
            print(f"Skipping missing file: {file_path}")
            continue

        samples = extract_raw_samples(file_path)

        if len(samples) == 0:
            print(f"No samples extracted from {file_path.name}")
            continue

        labels = np.full(len(samples), material)
        groups = np.full(len(samples), file_stem)

        X_parts.append(samples)
        y_parts.append(labels)
        group_parts.append(groups)

        print(
            f"{file_stem}: {len(samples)} samples, "
            f"label={get_material_name(material)}"
        )

    if not X_parts:
        raise RuntimeError("No stable material .Q00 files were found.")

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    groups = np.concatenate(group_parts)

    return X, y, groups


def create_model():
    """Create the KNN pipeline used for classification."""
    return make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(
            n_neighbors=2,
            weights="distance",
        ),
    )


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list,
    path: Path,
    title: str,
):
    """Save and show a confusion matrix."""
    readable_labels = [get_material_name(label) for label in labels]
    cm = confusion_matrix(y_true, y_pred, labels=labels)

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
    )

    ax.set_xlabel("Predicted material")
    ax.set_ylabel("True material")
    ax.set_title(title)

    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")


def plot_class_distribution(y: np.ndarray, path: Path):
    """Save and show the number of samples per material class."""
    unique_labels, counts = np.unique(y, return_counts=True)
    readable_labels = [get_material_name(label) for label in unique_labels]

    colors = {
        "Bismuth (Bi)": "#4C72B0",
        "Chromium (Cr)": "#DD8452",
        "Iron (Fe)": "#55A868",
    }

    bar_colors = [colors.get(label, "#8172B3") for label in readable_labels]

    plt.figure(figsize=(9, 6))
    bars = plt.bar(
        readable_labels,
        counts,
        color=bar_colors,
        edgecolor="black",
    )

    for bar, count in zip(bars, counts):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts) * 0.01,
            str(count),
            ha="center",
            va="bottom",
        )

    plt.xlabel("Material")
    plt.ylabel("Number of samples")
    plt.title("Class Distribution of Conductivity Fluctuation Samples")
    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")


def evaluate_random_split(X: np.ndarray, y: np.ndarray):
    """
    Evaluate KNN using a random sample-level train-test split.

    This split does not keep overlapping windows from the same file together,
    so its accuracy can be optimistically biased. It is retained only for
    comparison with the original implementation.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = create_model()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "macro_f1": f1_score(
            y_test,
            y_pred,
            labels=MATERIAL_ORDER,
            average="macro",
            zero_division=0,
        ),
    }

    print("\nRandom sample-level train-test split")
    print(f"Train samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Accuracy: {metrics['accuracy'] * 100:.2f}%")
    print(
        f"Balanced accuracy: {metrics['balanced_accuracy'] * 100:.2f}%"
    )
    print(f"Macro F1: {metrics['macro_f1']:.4f}")

    print(
        classification_report(
            y_test,
            y_pred,
            labels=MATERIAL_ORDER,
            target_names=[
                get_material_name(label) for label in MATERIAL_ORDER
            ],
            zero_division=0,
        )
    )

    plot_confusion_matrix(
        y_true=y_test,
        y_pred=y_pred,
        labels=MATERIAL_ORDER,
        path=FIGURES_DIR / "knn_raw_q00_random_split_confusion_matrix.png",
        title=(
            "KNN Classification of Thin-Film Materials\n"
            "Raw Q00 Samples — Random Sample-Level Split"
        ),
    )

    return metrics


def evaluate_file_level_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
):
    """
    Evaluate KNN using file-level GroupKFold cross-validation.

    All overlapping windows from one .Q00 file remain in the same fold.
    This is the primary, more conservative evaluation.
    """
    group_cv = GroupKFold(n_splits=N_SPLITS)

    cv_scores = cross_val_score(
        create_model(),
        X,
        y,
        groups=groups,
        cv=group_cv,
        scoring="accuracy",
        n_jobs=-1,
    )

    metrics = {
        "cv_mean_accuracy": cv_scores.mean(),
        "cv_std_accuracy": cv_scores.std(),
        "cv_fold_accuracies": cv_scores.tolist(),
    }

    print(f"\nFile-level {N_SPLITS}-fold GroupKFold cross-validation")
    for fold, score in enumerate(cv_scores, start=1):
        print(f"Fold {fold}: {score * 100:.2f}%")

    print(f"Mean accuracy: {metrics['cv_mean_accuracy'] * 100:.2f}%")
    print(f"Accuracy std: {metrics['cv_std_accuracy'] * 100:.2f}%")

    return metrics


def save_metrics(
    random_split_metrics: dict,
    file_cv_metrics: dict,
):
    """Save both evaluation protocols to separate CSV files."""
    random_split_df = pd.DataFrame(
        [
            {
                "model": "KNN — raw Q00 samples",
                "evaluation": "random_sample_split",
                "accuracy": random_split_metrics["accuracy"],
                "balanced_accuracy": random_split_metrics["balanced_accuracy"],
                "macro_f1": random_split_metrics["macro_f1"],
                "n_neighbors": 5,
                "sample_length": SAMPLE_LENGTH,
                "overlap_mode": OVERLAP_MODE,
                "test_size": TEST_SIZE,
                "random_state": RANDOM_STATE,
            }
        ]
    )

    random_split_path = (
        METRICS_DIR / "knn_raw_q00_random_split_metrics.csv"
    )
    random_split_df.to_csv(random_split_path, index=False)

    file_cv_df = pd.DataFrame(
        [
            {
                "model": "KNN — raw Q00 samples",
                "evaluation": "file_level_group_kfold",
                "cv_mean_accuracy": file_cv_metrics["cv_mean_accuracy"],
                "cv_std_accuracy": file_cv_metrics["cv_std_accuracy"],
                "cv_fold_accuracies": "; ".join(
                    f"{score:.6f}"
                    for score in file_cv_metrics["cv_fold_accuracies"]
                ),
                "n_splits": N_SPLITS,
                "n_neighbors": 5,
                "sample_length": SAMPLE_LENGTH,
                "overlap_mode": OVERLAP_MODE,
            }
        ]
    )

    file_cv_path = METRICS_DIR / "knn_raw_q00_file_cv_metrics.csv"
    file_cv_df.to_csv(file_cv_path, index=False)

    print(f"\nRandom-split metrics saved: {random_split_path}")
    print(f"File-level CV metrics saved: {file_cv_path}")


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    X, y, groups = build_dataset()

    print(f"\nDataset shape: {X.shape}")
    print(
        "Class distribution:",
        dict(zip(*np.unique(y, return_counts=True))),
    )
    print(f"Number of source files: {len(np.unique(groups))}")

    plot_class_distribution(
        y=y,
        path=FIGURES_DIR / "class_distribution.png",
    )

    random_split_metrics = evaluate_random_split(X, y)

    file_cv_metrics = evaluate_file_level_cv(
        X=X,
        y=y,
        groups=groups,
    )

    save_metrics(
        random_split_metrics=random_split_metrics,
        file_cv_metrics=file_cv_metrics,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()

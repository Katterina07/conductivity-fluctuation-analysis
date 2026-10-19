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


DATA_DIR = Path("../data/q03")
RESULTS_DIR = Path("../results")
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
            f"label={material}"
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
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )

    plt.xlabel("Predicted material")
    plt.ylabel("True material")
    plt.title("KNN confusion matrix — raw conductivity samples")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
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
            target_names=MATERIAL_ORDER,
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


if __name__ == "__main__":
    main()

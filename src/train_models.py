"""
Train and evaluate models on conductivity fluctuation spectral features.

Stable BI, CR, and FE films are used for supervised material recognition.

XXX is a modal reference film. It is excluded from supervised material
classification and used separately to analyze composition-independent noise.

Irradiated films are not used for training. They are analyzed separately
to assess possible radiation-induced changes in 1/f noise characteristics.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = BASE_DIR / "data" / "processed" / "conductivity_features.csv"
FIGURES_DIR = BASE_DIR / "results" / "figures"
METRICS_DIR = BASE_DIR / "results" / "metrics"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

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

RANDOM_STATE = 42


def material_name(label: str) -> str:
    return MATERIAL_NAMES.get(label, label)


def load_dataset() -> pd.DataFrame:
    """
    Load the combined spectral-feature dataset.
    """
    return pd.read_csv(DATASET_PATH)


def prepare_training_data(df: pd.DataFrame):
    """
    Prepare stable, non-irradiated material films for supervised learning.
    """
    df_train = df[
        (df["used_for_training"] == True)
        & (df["material"] != REFERENCE_LABEL)
        & (df["irradiated"] == False)
    ].copy()

    X = df_train[["gamma", "log_energy"]].copy()
    y = df_train["material"].copy()

    return X, y, df_train


def plot_clusters(
    X_scaled: np.ndarray,
    labels: np.ndarray,
    path: Path,
):
    """
    Save a K-Means cluster plot.
    """
    plt.figure(figsize=(10, 6.5))

    scatter = plt.scatter(
        X_scaled[:, 0],
        X_scaled[:, 1],
        c=labels,
        cmap="viridis",
        s=25,
        alpha=0.75,
        edgecolor="black",
        linewidth=0.2,
    )

    plt.xlabel("Spectral exponent γ (standardized)", fontsize=12)
    plt.ylabel("Log10 spectral energy (standardized)", fontsize=12)
    plt.title(
        "K-Means Clustering of Stable Film Noise Features",
        fontsize=14,
        pad=15,
    )

    cbar = plt.colorbar(scatter)
    cbar.set_label("Cluster ID", fontsize=11)

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(
    y_test,
    y_pred,
    labels,
    path: Path,
):
    """
    Save a confusion matrix with readable material names.
    """
    readable_labels = [material_name(label) for label in labels]

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
        "Neural Network Classification\nof Stable Thin-Film Materials",
        fontsize=14,
        pad=15,
    )

    plt.xticks(rotation=30, ha="right", fontsize=11)
    plt.yticks(rotation=0, fontsize=11)

    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_file_level_comparison(
    df: pd.DataFrame,
    feature: str,
    ylabel: str,
    filename: str,
):
    """
    Compare stable and irradiated films using file-level mean values.

    Error bars show one standard deviation within each file.
    """
    summary = (
        df[df["material"] != REFERENCE_LABEL]
        .groupby(["file", "material", "irradiated"], as_index=False)
        .agg(
            mean_value=(feature, "mean"),
            std_value=(feature, "std"),
            n_samples=(feature, "size"),
        )
    )

    summary["label"] = summary["file"] + "\n(" + summary["material"] + ")"

    summary = summary.sort_values(
        ["material", "irradiated", "file"],
        ascending=[True, True, True],
    )

    plt.figure(figsize=(13, 7))

    for material in MATERIAL_ORDER:
        material_summary = summary[summary["material"] == material]

        for _, row in material_summary.iterrows():
            marker = "o" if not row["irradiated"] else "X"
            edgecolor = (
                "black" if row["irradiated"] else MATERIAL_COLORS[material]
            )
            linewidth = 1.2 if row["irradiated"] else 0.8

            plt.errorbar(
                row["label"],
                row["mean_value"],
                yerr=row["std_value"],
                fmt=marker,
                markersize=10,
                color=MATERIAL_COLORS[material],
                markeredgecolor=edgecolor,
                markeredgewidth=linewidth,
                capsize=5,
                elinewidth=1.5,
                label=(
                    f"{material_name(material)} — "
                    + ("irradiated" if row["irradiated"] else "stable")
                ),
            )

    plt.ylabel(ylabel, fontsize=12)
    plt.xlabel("Measurement file", fontsize=12)
    plt.title(
        f"File-Level Comparison of {ylabel}\nStable vs Irradiated Films",
        fontsize=14,
        pad=15,
    )
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.grid(True, axis="y", alpha=0.3)

    handles, labels = plt.gca().get_legend_handles_labels()
    unique = dict(zip(labels, handles))

    plt.legend(
        unique.values(),
        unique.keys(),
        fontsize=9,
        title_fontsize=10,
        loc="best",
    )

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


def plot_feature_boxplots(
    df: pd.DataFrame,
    feature: str,
    ylabel: str,
    filename: str,
):
    """
    Compare the distribution of a spectral feature for each measurement file.
    """
    plot_df = df[df["material"] != REFERENCE_LABEL].copy()
    plot_df["file_label"] = (
        plot_df["file"] + "\n(" + plot_df["material"] + ")"
    )

    file_order = (
        plot_df[["file", "material", "irradiated"]]
        .drop_duplicates()
        .sort_values(["material", "irradiated", "file"])
    )

    ordered_labels = [
        f"{row['file']}\n({row['material']})"
        for _, row in file_order.iterrows()
    ]

    plt.figure(figsize=(15, 7))

    sns.boxplot(
        data=plot_df,
        x="file_label",
        y=feature,
        hue="irradiated",
        order=ordered_labels,
        palette={False: "#B0BEC5", True: "#FFB74D"},
        showfliers=False,
        width=0.7,
    )

    plt.ylabel(ylabel, fontsize=12)
    plt.xlabel("Measurement file", fontsize=12)
    plt.title(
        f"Distribution of {ylabel}\nby Measurement File and Irradiation State",
        fontsize=14,
        pad=15,
    )
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.grid(True, axis="y", alpha=0.3)

    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(
        handles,
        ["Stable", "Irradiated"],
        title="Film state",
        fontsize=10,
        title_fontsize=11,
    )

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


def analyze_irradiated_films(
    df: pd.DataFrame,
    model,
    scaler,
):
    """
    Apply the trained model to irradiated films and summarize
    radiation-induced changes at the file level.
    """
    df_irradiated = df[df["irradiated"] == True].copy()

    if df_irradiated.empty:
        print("No irradiated films found.")
        return df_irradiated

    X_irradiated = df_irradiated[["gamma", "log_energy"]].copy()
    X_irradiated_scaled = scaler.transform(X_irradiated)

    df_irradiated["predicted_material"] = model.predict(
        X_irradiated_scaled
    )

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_irradiated_scaled)
        df_irradiated["prediction_confidence"] = probabilities.max(axis=1)

    summary = (
        df_irradiated
        .groupby(["file", "material"], as_index=False)
        .agg(
            n_samples=("gamma", "size"),
            gamma_mean=("gamma", "mean"),
            gamma_std=("gamma", "std"),
            log_energy_mean=("log_energy", "mean"),
            log_energy_std=("log_energy", "std"),
            predicted_material=(
                "predicted_material",
                lambda x: x.mode().iloc[0],
            ),
            mean_prediction_confidence=(
                "prediction_confidence",
                "mean",
            ),
        )
    )

    summary["prediction_agreement"] = (
        summary["material"] == summary["predicted_material"]
    )

    print("\nIrradiated film analysis (file-level summary):")
    print(
        summary[
            [
                "file",
                "material",
                "n_samples",
                "gamma_mean",
                "gamma_std",
                "log_energy_mean",
                "log_energy_std",
                "predicted_material",
                "mean_prediction_confidence",
                "prediction_agreement",
            ]
        ]
        .to_string(index=False)
    )

    return df_irradiated


def main():
    df = load_dataset()

    X, y, df_train = prepare_training_data(df)

    print(f"Training samples: {len(X)}")
    print(
        "Class distribution:",
        y.value_counts().to_dict(),
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # -----------------------------
    # K-Means clustering
    # -----------------------------
    n_clusters = y.nunique()

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=RANDOM_STATE,
        n_init=20,
    )

    cluster_labels = kmeans.fit_predict(X_train_scaled)

    silhouette = silhouette_score(X_train_scaled, cluster_labels)

    print(f"K-Means silhouette score: {silhouette:.4f}")

    plot_clusters(
        X_train_scaled,
        cluster_labels,
        FIGURES_DIR / "kmeans_clusters.png",
    )

    # -----------------------------
    # KNN classification
    # -----------------------------
    knn = KNeighborsClassifier(
        n_neighbors=5,
        weights="distance",
    )

    knn.fit(X_train_scaled, y_train)

    knn_pred = knn.predict(X_test_scaled)
    knn_accuracy = accuracy_score(y_test, knn_pred)

    print(f"KNN accuracy: {knn_accuracy * 100:.2f}%")
    print(
        classification_report(
            y_test,
            knn_pred,
            labels=MATERIAL_ORDER,
            target_names=[
                material_name(label) for label in MATERIAL_ORDER
            ],
            zero_division=0,
        )
    )

    # -----------------------------
    # Fully connected neural network
    # -----------------------------
    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        max_iter=1000,
        early_stopping=True,
        random_state=RANDOM_STATE,
    )

    mlp.fit(X_train_scaled, y_train)

    mlp_pred = mlp.predict(X_test_scaled)
    mlp_accuracy = accuracy_score(y_test, mlp_pred)

    print(f"Neural network accuracy: {mlp_accuracy * 100:.2f}%")
    print(
        classification_report(
            y_test,
            mlp_pred,
            labels=MATERIAL_ORDER,
            target_names=[
                material_name(label) for label in MATERIAL_ORDER
            ],
            zero_division=0,
        )
    )

    plot_confusion_matrix(
        y_test=y_test,
        y_pred=mlp_pred,
        labels=MATERIAL_ORDER,
        path=FIGURES_DIR / "neural_network_confusion_matrix.png",
    )

    # -----------------------------
    # Irradiated film analysis
    # -----------------------------
    df_irradiated = analyze_irradiated_films(
        df=df,
        model=mlp,
        scaler=scaler,
    )

    # -----------------------------
    # File-level comparison plots
    # -----------------------------
    plot_file_level_comparison(
        df=df,
        feature="gamma",
        ylabel="Spectral exponent γ",
        filename="file_level_gamma_comparison.png",
    )

    plot_file_level_comparison(
        df=df,
        feature="log_energy",
        ylabel="Log10 spectral energy",
        filename="file_level_energy_comparison.png",
    )

    plot_feature_boxplots(
        df=df,
        feature="gamma",
        ylabel="Spectral exponent γ",
        filename="gamma_boxplots_by_file.png",
    )

    plot_feature_boxplots(
        df=df,
        feature="log_energy",
        ylabel="Log10 spectral energy",
        filename="energy_boxplots_by_file.png",
    )

    # -----------------------------
    # Save metrics
    # -----------------------------
    metrics = pd.DataFrame(
        [
            {
                "model": "K-Means",
                "metric": "silhouette_score",
                "value": silhouette,
            },
            {
                "model": "KNN",
                "metric": "accuracy",
                "value": knn_accuracy,
            },
            {
                "model": "Neural network",
                "metric": "accuracy",
                "value": mlp_accuracy,
            },
        ]
    )

    metrics.to_csv(
        METRICS_DIR / "model_metrics.csv",
        index=False,
    )

    if not df_irradiated.empty:
        df_irradiated.to_csv(
            METRICS_DIR / "irradiated_film_predictions.csv",
            index=False,
        )

    print(f"Metrics saved: {METRICS_DIR / 'model_metrics.csv'}")
    print(f"Figures saved: {FIGURES_DIR}")


if __name__ == "__main__":
    main()

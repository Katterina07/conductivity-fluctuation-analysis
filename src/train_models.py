"""
Train material-recognition models on stable films and analyze
radiation-induced changes in irradiated films.
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


def material_name(label):
    return MATERIAL_NAMES.get(label, label)


def load_dataset():
    return pd.read_csv(DATASET_PATH)


def prepare_training_data(df):
    """
    Use only stable, non-irradiated material films for supervised learning.
    """
    df_train = df[
        (df["used_for_training"] == True)
        & (df["material"] != REFERENCE_LABEL)
    ].copy()

    X = df_train[["gamma", "log_energy"]].copy()
    y = df_train["material"].copy()

    return X, y, df_train


def plot_material_scatter(df_train, df_irradiated, path):
    plt.figure(figsize=(11, 7))

    for material in MATERIAL_ORDER:
        subset = df_train[df_train["material"] == material]

        plt.scatter(
            subset["gamma"],
            subset["log_energy"],
            label=f"{material_name(material)} — training",
            color=MATERIAL_COLORS[material],
            alpha=0.8,
            s=70,
            edgecolor="black",
            linewidth=0.5,
        )

    for material in MATERIAL_ORDER:
        subset = df_irradiated[df_irradiated["material"] == material]

        plt.scatter(
            subset["gamma"],
            subset["log_energy"],
            label=f"{material_name(material)} — irradiated",
            color=MATERIAL_COLORS[material],
            marker="X",
            s=150,
            edgecolor="black",
            linewidth=0.8,
        )

    plt.xlabel("Spectral exponent γ", fontsize=12)
    plt.ylabel("Log10 spectral energy", fontsize=12)
    plt.title(
        "1/f Noise Features:\nStable Training Films vs Irradiated Films",
        fontsize=14,
        pad=15,
    )
    plt.legend(title="Material / state", fontsize=9, title_fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_clusters(X_scaled, labels, path):
    plt.figure(figsize=(10, 6.5))

    scatter = plt.scatter(
        X_scaled[:, 0],
        X_scaled[:, 1],
        c=labels,
        cmap="viridis",
        s=70,
        alpha=0.85,
        edgecolor="black",
        linewidth=0.4,
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


def plot_confusion_matrix(y_test, y_pred, labels, path):
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
    )

    ax.set_xlabel("Predicted material", fontsize=12)
    ax.set_ylabel("True material", fontsize=12)
    ax.set_title(
        "Neural Network Classification\nof Stable Thin-Film Materials",
        fontsize=14,
        pad=15,
    )

    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_irradiated_comparison(df, feature, xlabel, filename):
    plt.figure(figsize=(11, 6.5))

    for material in MATERIAL_ORDER:
        stable = df[
            (df["material"] == material)
            & (df["irradiated"] == False)
            & (df["used_for_training"] == True)
        ]

        irradiated_05 = df[
            (df["material"] == material)
            & (df["irradiated"] == True)
            & (df["file"].str.contains("105"))
        ]

        irradiated_06 = df[
            (df["material"] == material)
            & (df["irradiated"] == True)
            & (df["file"].str.contains("106"))
        ]

        plt.scatter(
            stable[feature],
            [material_name(material)] * len(stable),
            label=f"{material_name(material)} — stable",
            color=MATERIAL_COLORS[material],
            alpha=0.8,
            s=90,
            edgecolor="black",
        )

        plt.scatter(
            irradiated_05[feature],
            [material_name(material)] * len(irradiated_05),
            marker="X",
            s=160,
            label=f"{material_name(material)} — after irradiation, period 1",
            color=MATERIAL_COLORS[material],
            edgecolor="black",
        )

        plt.scatter(
            irradiated_06[feature],
            [material_name(material)] * len(irradiated_06),
            marker="P",
            s=170,
            label=f"{material_name(material)} — after irradiation, period 2",
            color=MATERIAL_COLORS[material],
            edgecolor="black",
            alpha=0.75,
        )

    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel("Material", fontsize=12)
    plt.title(
        f"Effect of Electromagnetic Pulse Irradiation on {xlabel}",
        fontsize=14,
        pad=15,
    )
    plt.legend(fontsize=8, title_fontsize=10, loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


def analyze_irradiated_films(df, model, scaler):
    """
    Compare irradiated films with the trained stable-film model.
    """
    df_irradiated = df[df["irradiated"] == True].copy()

    if df_irradiated.empty:
        print("No irradiated films found.")
        return df_irradiated

    X_irradiated = df_irradiated[["gamma", "log_energy"]].copy()
    X_irradiated_scaled = scaler.transform(X_irradiated)

    df_irradiated["predicted_material"] = model.predict(X_irradiated_scaled)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_irradiated_scaled)
        df_irradiated["prediction_confidence"] = probabilities.max(axis=1)

    print("\nIrradiated film analysis:")
    print(
        df_irradiated[
            [
                "file",
                "material",
                "gamma",
                "log_energy",
                "predicted_material",
                "prediction_confidence",
            ]
        ].to_string(index=False)
    )

    return df_irradiated


def main():
    df = load_dataset()

    X, y, df_train = prepare_training_data(df)

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

    # K-Means
    kmeans = KMeans(
        n_clusters=y.nunique(),
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

    # KNN
    knn = KNeighborsClassifier(
        n_neighbors=3,
        weights="distance",
    )

    knn.fit(X_train_scaled, y_train)
    knn_pred = knn.predict(X_test_scaled)
    knn_accuracy = accuracy_score(y_test, knn_pred)

    print(f"KNN accuracy: {knn_accuracy * 100:.2f}%")

    # Neural network
    mlp = MLPClassifier(
        hidden_layer_sizes=(32,),
        activation="relu",
        solver="lbfgs",
        alpha=1e-4,
        max_iter=2000,
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
            target_names=[material_name(label) for label in MATERIAL_ORDER],
            zero_division=0,
        )
    )

    plot_confusion_matrix(
        y_test,
        mlp_pred,
        MATERIAL_ORDER,
        FIGURES_DIR / "neural_network_confusion_matrix.png",
    )

    # Irradiated films
    df_irradiated = analyze_irradiated_films(df, mlp, scaler)

    plot_material_scatter(
        df_train,
        df_irradiated,
        FIGURES_DIR / "stable_vs_irradiated_films.png",
    )

    plot_irradiated_comparison(
        df,
        "gamma",
        "Spectral exponent γ",
        "irradiated_gamma_comparison.png",
    )

    plot_irradiated_comparison(
        df,
        "log_energy",
        "Log10 spectral energy",
        "irradiated_energy_comparison.png",
    )

    # Metrics
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

    metrics.to_csv(METRICS_DIR / "model_metrics.csv", index=False)

    if not df_irradiated.empty:
        df_irradiated.to_csv(
            METRICS_DIR / "irradiated_film_predictions.csv",
            index=False,
        )

    print(f"Metrics saved: {METRICS_DIR / 'model_metrics.csv'}")
    print(f"Figures saved: {FIGURES_DIR}")


if __name__ == "__main__":
    main()

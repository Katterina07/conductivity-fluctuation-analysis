"""
Train and evaluate models on conductivity fluctuation spectral features.

Material classification uses BI, CR, and FE films.

XXX is a modal reference film. It is excluded from supervised material
classification and used separately to analyze composition-independent noise.
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
RESULTS_DIR = Path("../results")
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"

REFERENCE_LABEL = "XXX"
MATERIAL_ORDER = ["BI", "CR", "FE"]

RANDOM_STATE = 42


def load_dataset():
    """
    Load the combined feature dataset.
    """
    return pd.read_csv(DATASET_PATH)


def prepare_material_dataset(df):
    """
    Prepare a dataset for supervised material classification.

    XXX reference samples are excluded because they do not represent
    a target material class.
    """
    df_materials = df[df["material"] != REFERENCE_LABEL].copy()

    X = df_materials[["gamma", "energy"]].copy()
    y = df_materials["material"].copy()

    # Energy can vary by several orders of magnitude
    X["energy"] = np.log10(X["energy"].clip(lower=1e-30))

    return X, y


def plot_clusters(X_scaled, labels, path):
    """
    Save a K-Means cluster plot.
    """
    plt.figure(figsize=(9, 6))

    sns.scatterplot(
        x=X_scaled[:, 0],
        y=X_scaled[:, 1],
        hue=labels,
        palette="tab10",
        s=25,
    )

    plt.xlabel("Gamma (scaled)")
    plt.ylabel("Log10 energy (scaled)")
    plt.title("K-Means clustering of conductivity fluctuation features")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_confusion_matrix(y_test, y_pred, labels, path):
    """
    Save a confusion matrix.
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
    plt.title("Neural network confusion matrix")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def analyze_reference_film(df):
    """
    Compare the modal reference film with material films.

    XXX helps identify noise components that are not related
    to the chemical composition of the films.
    """
    df_reference = df[df["material"] == REFERENCE_LABEL].copy()
    df_materials = df[df["material"] != REFERENCE_LABEL].copy()

    print(f"Reference film samples: {len(df_reference)}")
    print(f"Material film samples: {len(df_materials)}")

    # Gamma distribution
    plt.figure(figsize=(10, 6))

    sns.kdeplot(
        data=df_materials,
        x="gamma",
        hue="material",
        common_norm=False,
        fill=True,
        alpha=0.25,
    )

    sns.kdeplot(
        data=df_reference,
        x="gamma",
        color="black",
        linewidth=2,
        label="XXX reference film",
    )

    plt.xlabel("Gamma")
    plt.ylabel("Density")
    plt.title("Gamma distribution: material films vs reference film")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "reference_vs_materials_gamma.png",
        dpi=300,
    )
    plt.close()

    # Energy distribution
    df_materials_plot = df_materials.copy()
    df_reference_plot = df_reference.copy()

    df_materials_plot["log_energy"] = np.log10(
        df_materials_plot["energy"].clip(lower=1e-30)
    )

    df_reference_plot["log_energy"] = np.log10(
        df_reference_plot["energy"].clip(lower=1e-30)
    )

    plt.figure(figsize=(10, 6))

    sns.kdeplot(
        data=df_materials_plot,
        x="log_energy",
        hue="material",
        common_norm=False,
        fill=True,
        alpha=0.25,
    )

    sns.kdeplot(
        data=df_reference_plot,
        x="log_energy",
        color="black",
        linewidth=2,
        label="XXX reference film",
    )

    plt.xlabel("Log10 spectral energy")
    plt.ylabel("Density")
    plt.title("Energy distribution: material films vs reference film")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "reference_vs_materials_energy.png",
        dpi=300,
    )
    plt.close()

    reference_summary = df_reference[["gamma", "energy"]].describe()
    material_summary = df_materials.groupby("material")[
        ["gamma", "energy"]
    ].describe()

    reference_summary.to_csv(
        METRICS_DIR / "reference_film_summary.csv"
    )

    material_summary.to_csv(
        METRICS_DIR / "material_film_summary.csv"
    )


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset()

    analyze_reference_film(df)

    X, y = prepare_material_dataset(df)

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
            target_names=MATERIAL_ORDER,
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
            target_names=MATERIAL_ORDER,
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

    print(f"Metrics saved: {METRICS_DIR / 'model_metrics.csv'}")


if __name__ == "__main__":
    main()

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, balanced_accuracy_score

BASE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = BASE_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from train_knn_raw_samples import (
    build_dataset,
    create_model,
    get_material_name,
)

X, y, groups = build_dataset()

cv = LeaveOneGroupOut()
rows = []

for fold, (train_idx, test_idx) in enumerate(
    cv.split(X, y, groups), start=1
):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    test_file = np.unique(groups[test_idx])[0]

    model = create_model()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    balanced = balanced_accuracy_score(y_test, y_pred)

    print(f"Fold {fold}: held-out file {test_file}")
    print(f"True material: {y_test[0]}")
    print(f"Predicted classes: {dict(zip(*np.unique(y_pred, return_counts=True)))}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("-" * 50)

    rows.append(
        {
            "held_out_file": test_file,
            "true_material": y_test[0],
            "accuracy": accuracy,
            "balanced_accuracy": balanced,
            "n_test_samples": len(X_test),
        }
    )

results = pd.DataFrame(rows)
results.loc[len(results)] = {
    "held_out_file": "MEAN",
    "true_material": "",
    "accuracy": results["accuracy"].mean(),
    "balanced_accuracy": results["balanced_accuracy"].mean(),
    "n_test_samples": results["n_test_samples"].sum(),
}

print(results.to_string(index=False))

output = BASE_DIR / "results" / "metrics" / "knn_raw_q00_leave_one_file_out_metrics.csv"
output.parent.mkdir(parents=True, exist_ok=True)
results.to_csv(output, index=False)
print(f"Saved: {output}")

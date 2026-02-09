import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import RocCurveDisplay, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split


def main() -> None:
    Path("reports").mkdir(parents=True, exist_ok=True)
    Path("figures").mkdir(parents=True, exist_ok=True)

    ds = load_breast_cancer(as_frame=True)
    X: pd.DataFrame = ds.data
    y = ds.target  # 0/1 labels

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=5000, solver="liblinear")
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "dataset": "sklearn.datasets.load_breast_cancer",
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "test_size": 0.20,
        "random_state": 42,
        "model": "LogisticRegression(solver=liblinear)",
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
    }

    with open("reports/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_test, y_prob, ax=ax)
    ax.set_title("ROC curve (baseline model)")
    fig.tight_layout()
    fig.savefig("figures/roc_curve.png", dpi=200)

    print("Saved reports/metrics.json and figures/roc_curve.png")


if __name__ == "__main__":
    main()

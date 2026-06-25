from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def evaluate_models(models: dict[str, Any], X_test: Any, y_test: Any, feature_names: list[str], output_dir: Path) -> dict[str, dict[str, float]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, float]] = {}

    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
        f1 = f1_score(y_test, y_pred)

        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Matrice de confusion - {name}")
        plt.xlabel("Prédit")
        plt.ylabel("Réel")
        plt.tight_layout()
        plt.savefig(output_dir / f"confusion_{name}.png", dpi=180)
        plt.close()

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.figure(figsize=(6, 4))
        plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.title(f"ROC Curve - {name}")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"roc_{name}.png", dpi=180)
        plt.close()

        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        plt.figure(figsize=(6, 4))
        plt.plot(recall, precision, label=f"PR-AUC = {pr_auc:.3f}")
        plt.title(f"Precision-Recall - {name}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"pr_{name}.png", dpi=180)
        plt.close()

        results[name] = {
            "accuracy": float(accuracy),
            "auc": float(auc),
            "pr_auc": float(pr_auc),
            "f1": float(f1),
        }

        if hasattr(model, "feature_importances_") and feature_names:
            importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
            plt.figure(figsize=(8, 5))
            importances.head(15).plot(kind="bar")
            plt.title(f"Feature Importance - {name}")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(output_dir / f"feature_importance_{name}.png", dpi=180)
            plt.close()

    metrics_df = pd.DataFrame(results).T
    metrics_df.index.name = "model"
    metrics_df = metrics_df[["accuracy", "auc", "pr_auc", "f1"]]
    metrics_df.to_csv(output_dir / "metrics_summary.csv")

    plt.figure(figsize=(10, 6))
    sns.heatmap(metrics_df.round(3), annot=True, cmap="viridis", fmt=".3f")
    plt.title("Résumé des métriques par modèle")
    plt.tight_layout()
    plt.savefig(output_dir / "metrics_comparison.png", dpi=180)
    plt.close()

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    return results

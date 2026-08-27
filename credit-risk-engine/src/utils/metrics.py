"""
metrics.py
==========

Credit-risk-specific evaluation utilities.

Responsibility
--------------
Centralizes the evaluation metrics and plots used across every model
trained in this project (baseline and future candidates), so that
all models are compared against exactly the same yardstick.

Beyond standard classification metrics, this module implements the
Kolmogorov-Smirnov (KS) statistic, the de-facto industry standard
for scorecard evaluation in credit risk: it measures the maximum
separation between the cumulative distributions of "good" and "bad"
borrowers across score deciles.
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def ks_statistic(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """
    Kolmogorov-Smirnov statistic: max distance between the cumulative
    distribution of predicted probabilities for the positive class (default)
    and the negative class (non-default). Values above ~0.30-0.40 are
    generally considered acceptable in credit scoring; above 0.50, strong.
    """
    df = pd.DataFrame({"y_true": y_true, "y_proba": y_proba})
    df = df.sort_values("y_proba")

    df["cum_bad"] = (df["y_true"] == 1).cumsum() / (df["y_true"] == 1).sum()
    df["cum_good"] = (df["y_true"] == 0).cumsum() / (df["y_true"] == 0).sum()

    ks = np.max(np.abs(df["cum_bad"] - df["cum_good"]))
    return float(ks)


def evaluate_classifier(y_true, y_pred, y_proba) -> dict:
    """Computes the full metric set for a binary credit-default classifier."""
    metrics = {
        "auc_roc": float(roc_auc_score(y_true, y_proba)),
        "ks_statistic": ks_statistic(np.asarray(y_true), np.asarray(y_proba)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1_score": float(f1_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, output_dict=True),
    }

    logging.info(f"AUC-ROC:   {metrics['auc_roc']:.4f}")
    logging.info(f"KS stat:   {metrics['ks_statistic']:.4f}")
    logging.info(f"Precision: {metrics['precision']:.4f}")
    logging.info(f"Recall:    {metrics['recall']:.4f}")
    logging.info(f"F1-score:  {metrics['f1_score']:.4f}")
    logging.info(f"Confusion matrix: {metrics['confusion_matrix']}")

    return metrics


def plot_roc_curve(y_true, y_proba, save_path: str, model_name: str = "Model"):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random baseline")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve — {model_name}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logging.info(f"ROC curve saved to {save_path}")


def plot_confusion_matrix(cm, save_path: str, model_name: str = "Model"):
    cm = np.array(cm)
    plt.figure(figsize=(5, 5))
    plt.imshow(cm, cmap="Blues")
    plt.title(f"Confusion Matrix — {model_name}")
    plt.colorbar()
    ticks = ["No Default", "Default"]
    plt.xticks([0, 1], ticks)
    plt.yticks([0, 1], ticks)
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center",
                      color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logging.info(f"Confusion matrix plot saved to {save_path}")
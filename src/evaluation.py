"""
evaluation.py — Métricas, limiar de decisão e gráficos de avaliação.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from . import config as cfg


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------
def compute_metrics(y_true, y_proba, threshold: float = 0.5) -> dict:
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "brier": float(brier_score_loss(y_true, y_proba)),
    }


def best_threshold(y_true, y_proba, criterion: str = "f1") -> float:
    """
    Escolhe o limiar de decisão.

    IMPORTANTE: deve ser calibrado em validação cruzada sobre o TREINO. Ajustar
    o limiar olhando o teste é vazamento — o teste passaria a participar da
    escolha de um hiperparâmetro.
    """
    prec, rec, thr = precision_recall_curve(y_true, y_proba)
    if criterion == "f1":
        f1 = np.divide(
            2 * prec * rec, prec + rec, out=np.zeros_like(prec), where=(prec + rec) > 0
        )
        idx = int(np.nanargmax(f1[:-1])) if len(thr) else 0
        return float(thr[idx]) if len(thr) else 0.5
    if criterion == "youden":
        fpr, tpr, t = roc_curve(y_true, y_proba)
        return float(t[int(np.argmax(tpr - fpr))])
    raise ValueError(criterion)


def metrics_table(rows: dict[str, dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows).T
    order = [
        "roc_auc", "pr_auc", "accuracy", "balanced_accuracy",
        "precision", "recall", "f1", "mcc", "brier", "threshold",
    ]
    return df[[c for c in order if c in df.columns]]


def save_json(obj, path) -> None:
    def _default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False, default=_default)


def text_report(y_true, y_proba, threshold: float) -> str:
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    rep = classification_report(
        y_true, y_pred, target_names=["Não sobreviveu", "Sobreviveu"], digits=3
    )
    return f"Matriz de confusão:\n{cm}\n\n{rep}"

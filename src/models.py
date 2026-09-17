"""
models.py — Catálogo de modelos candidatos e respectivas grades de busca.

Cada candidato é um Pipeline completo (pré-processamento + estimador), de forma
que a busca de hiperparâmetros com validação cruzada refaz o pré-processamento
em cada fold. Passar uma matriz já transformada para o GridSearchCV seria
vazamento de pré-processamento.
"""

from __future__ import annotations

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from . import config as cfg
from .features import build_preprocessor


def _pipe(estimator, scale_numeric: bool) -> Pipeline:
    return Pipeline(
        [
            ("prep", build_preprocessor(scale_numeric=scale_numeric)),
            ("clf", estimator),
        ]
    )


def get_candidates() -> dict[str, dict]:
    """
    Devolve {nome: {'pipeline': Pipeline, 'grid': dict, 'note': str}}.

    Justificativa das escolhas:
      * Baseline        — piso de referência; qualquer modelo precisa superá-lo.
      * Árvore          — modelo simples e interpretável; serve para mostrar o
                          efeito da profundidade sobre overfitting.
      * Reg. Logística  — linear, calibrada, coeficientes lidos como razão de
                          chances; exige padronização e encoding.
      * Random Forest   — bagging; robusto, baixa variância, pouco tuning.
      * HistGB          — boosting histogramado; estado da arte em dados
                          tabulares e o candidato esperado a vencer.
    """
    bal = "balanced"   # compensa o desbalanceamento 62/38 sem reamostrar

    return {
        "Baseline (classe majoritária)": {
            "pipeline": _pipe(DummyClassifier(strategy="prior"), scale_numeric=False),
            "grid": {},
            "note": "piso de referência",
        },
        "Árvore de Decisão": {
            "pipeline": _pipe(
                DecisionTreeClassifier(random_state=cfg.RANDOM_STATE, class_weight=bal),
                scale_numeric=False,
            ),
            "grid": {
                "clf__max_depth": [3, 4, 5, 6, 8, None],
                "clf__min_samples_leaf": [1, 5, 10, 20, 40],
                "clf__criterion": ["gini", "entropy"],
            },
            "note": "interpretável; controle direto de complexidade",
        },
        "Regressão Logística": {
            "pipeline": _pipe(
                LogisticRegression(
                    max_iter=5000,
                    solver="liblinear",
                    class_weight=bal,
                    random_state=cfg.RANDOM_STATE,
                ),
                scale_numeric=True,
            ),
            "grid": {
                "clf__C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
                "clf__penalty": ["l1", "l2"],
            },
            "note": "linear regularizada; L1 lida com a colinearidade dos agregados",
        },
        "Random Forest": {
            "pipeline": _pipe(
                RandomForestClassifier(
                    random_state=cfg.RANDOM_STATE,
                    class_weight="balanced_subsample",
                    n_jobs=1,   # paralelismo fica no GridSearchCV (evita aninhamento)
                ),
                scale_numeric=False,
            ),
            "grid": {
                "clf__n_estimators": [300],
                "clf__max_depth": [6, 10],
                "clf__min_samples_leaf": [8, 15],
                "clf__max_features": ["sqrt", 0.4],
            },
            "note": "bagging; variância baixa, pouco sensível a hiperparâmetros",
        },
        "Gradient Boosting (HistGB)": {
            "pipeline": _pipe(
                HistGradientBoostingClassifier(
                    random_state=cfg.RANDOM_STATE,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=25,
                    class_weight=bal,
                ),
                scale_numeric=False,
            ),
            "grid": {
                "clf__learning_rate": [0.03, 0.06, 0.1],
                "clf__max_leaf_nodes": [7, 15, 31],
                "clf__min_samples_leaf": [10, 20, 40],
                "clf__l2_regularization": [0.0, 1.0],
                "clf__max_iter": [300],
            },
            "note": "boosting de árvores; referência em dados tabulares",
        },
    }

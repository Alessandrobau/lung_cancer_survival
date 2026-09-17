"""
data.py — Carga, limpeza e divisão dos dados.

Ponto único de entrada para obter (X, y). A remoção de colunas com vazamento
acontece AQUI, antes de qualquer divisão, para que nenhuma etapa posterior
possa acidentalmente reintroduzi-las.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from . import config as cfg
from . import validate


def load_raw(path=None, validate_data: bool = True, verbose: bool = True) -> pd.DataFrame:
    """Carrega o CSV bruto e roda o harness de validação."""
    df = pd.read_csv(path or cfg.DATA_RAW)
    if validate_data:
        validate.validate_raw(df, verbose=verbose)
    return df


def make_xy(
    df: pd.DataFrame,
    include_leaky: bool = False,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Constrói a matriz de features e o vetor alvo.

    Parameters
    ----------
    include_leaky : bool
        Se True, MANTÉM propositalmente as colunas com vazamento
        (Survival_Months, Treatment). Usado apenas no experimento de
        quantificação de vazamento (src/leakage_experiment.py). O modelo de
        produção sempre usa include_leaky=False.
    """
    y = df[cfg.TARGET].map(cfg.TARGET_MAP).astype(int)

    drop = list(cfg.DROP_ALWAYS)
    if include_leaky:
        drop = [c for c in drop if c not in (cfg.LEAKY_OUTCOME + cfg.LEAKY_POSTERIOR)]

    X = df.drop(columns=[cfg.TARGET] + drop, errors="ignore")

    if not include_leaky:
        validate.validate_no_leakage(X, verbose=verbose)

    return X, y


def split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = cfg.TEST_SIZE,
    stratify: bool = True,
    verbose: bool = True,
):
    """Divisão hold-out estratificada, com validação de disjunção."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=cfg.RANDOM_STATE,
        stratify=y if stratify else None,
        shuffle=True,
    )
    validate.validate_split(X_tr, X_te, verbose=verbose)
    return X_tr, X_te, y_tr, y_te


def temporal_split(df: pd.DataFrame, cutoff: int = cfg.TEMPORAL_CUTOFF_YEAR):
    """
    Divisão temporal por ano de diagnóstico (teste de robustez).

    Simula o uso real do modelo: treinar com o histórico disponível e prever
    pacientes futuros. Mais pessimista e mais honesta que o hold-out aleatório
    quando existe deriva temporal.
    """
    X, y = make_xy(df, verbose=False)
    is_train = df["Diagnosis_Year"] < cutoff
    return X[is_train], X[~is_train], y[is_train], y[~is_train]

"""
features.py — Transformação de atributos.

Todo o pré-processamento é construído como um ColumnTransformer que vive DENTRO
do Pipeline do scikit-learn. Isso é o que impede vazamento de pré-processamento:
médias, desvios, medianas e vocabulários de categorias são aprendidos apenas nos
folds de treino e depois aplicados ao fold de validação / ao conjunto de teste.

Aprender o scaler ou o encoder no dataset inteiro antes do split é o erro mais
comum de vazamento em projetos de ML, e é exatamente o que este desenho evita.
"""

from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

from . import config as cfg

# Colunas que recebem tratamento ordinal (existe ordem clínica real)
ORDINAL_SPEC = {
    "Cancer_Stage": cfg.STAGE_ORDER,
    "Air_Pollution_Exposure": ["Low", "Moderate", "High"],
    "Alcohol_Use": ["No Alcohol", "Moderate", "Heavy"],
    "Exercise_Frequency": ["Low", "Moderate", "High"],
    "Smoking_Status": ["Never Smoked", "Former Smoker", "Current Smoker"],
}

# Categóricas nominais restantes -> One-Hot
NOMINAL = [
    c for c in cfg.CATEGORICAL_FEATURES if c not in ORDINAL_SPEC
]  # WHO_Region, Gender, Genetic_Mutation, Cancer_Type, NSCLC_Subtype, Diagnosis_Method


def _binary_encoder():
    """Yes/No -> 1/0, com categorias fixadas para não depender do fold."""
    return OrdinalEncoder(
        categories=[["No", "Yes"]] * len(cfg.BINARY_FEATURES),
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        dtype=np.float64,
    )


def build_preprocessor(
    scale_numeric: bool = True,
    extra_numeric: list[str] | None = None,
    extra_nominal: list[str] | None = None,
) -> ColumnTransformer:
    """
    Monta o pré-processador.

    Parameters
    ----------
    scale_numeric : bool
        True  -> padroniza numéricas (obrigatório p/ Regressão Logística e kNN).
        False -> mantém escala original (árvores são invariantes a escala
                 monotônica; padronizar só adiciona custo).
    extra_numeric, extra_nominal : list[str] | None
        Colunas adicionais. Usadas EXCLUSIVAMENTE pelo experimento de
        quantificação de vazamento (src/leakage_experiment.py), nunca pelo
        modelo de produção.
    """
    numeric_cols = list(cfg.NUMERIC_FEATURES) + list(extra_numeric or [])
    nominal_cols = list(NOMINAL) + list(extra_nominal or [])

    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipe = Pipeline(numeric_steps)

    binary_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", _binary_encoder()),
        ]
    )

    ordinal_cols = list(ORDINAL_SPEC)
    ordinal_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(
                    categories=[ORDINAL_SPEC[c] for c in ordinal_cols],
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    dtype=np.float64,
                ),
            ),
        ]
    )
    if scale_numeric:
        ordinal_pipe.steps.append(("scaler", StandardScaler()))

    nominal_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=10,      # agrupa níveis raros -> evita overfit
                    sparse_output=False,
                    drop=None,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("bin", binary_pipe, cfg.BINARY_FEATURES),
            ("ord", ordinal_pipe, ordinal_cols),
            ("nom", nominal_pipe, nominal_cols),
        ],
        remainder="drop",          # qualquer coluna não declarada é descartada
        verbose_feature_names_out=False,
    )


def feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Nomes das colunas após a transformação (para SHAP e importâncias)."""
    return list(preprocessor.get_feature_names_out())

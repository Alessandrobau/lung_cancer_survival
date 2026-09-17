"""
validate.py — Harness de validação de dados.

Conjunto de regras executáveis que falham alto (AssertionError) quando alguma
premissa do projeto deixa de valer. É executado no início de todo treinamento,
de modo que nenhum modelo é treinado sobre dados que violem o contrato.

Regras implementadas
--------------------
R1  schema      : todas as colunas esperadas existem e têm o tipo certo
R2  alvo        : binário, sem ausentes, com as duas classes presentes
R3  ausentes    : nenhum valor nulo nas colunas usadas
R4  duplicidade : nenhum paciente duplicado (com ou sem Patient_ID)
R5  domínio     : numéricas dentro de faixas clinicamente plausíveis
R6  categorias  : níveis observados pertencem ao vocabulário esperado
R7  coerência   : regras de negócio (não fumante => 0 cigarros, etc.)
R8  vazamento   : nenhuma coluna proibida sobrevive ao pré-processamento
R9  derivação   : colunas agregadas continuam sendo função exata das origens
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg


class ValidationError(AssertionError):
    """Erro de validação do contrato de dados."""


def _check(condition: bool, message: str, report: list[str]) -> None:
    if condition:
        report.append(f"  [OK]    {message}")
    else:
        report.append(f"  [FALHA] {message}")
        raise ValidationError(message)


# ---------------------------------------------------------------------------
# R1-R7: validação do dataframe bruto
# ---------------------------------------------------------------------------
def validate_raw(df: pd.DataFrame, verbose: bool = True) -> list[str]:
    report: list[str] = ["VALIDACAO DO DATASET BRUTO"]

    # R1 — schema
    expected = (
        [cfg.TARGET]
        + cfg.NUMERIC_FEATURES
        + cfg.BINARY_FEATURES
        + cfg.CATEGORICAL_FEATURES
        + cfg.DROP_ALWAYS
    )
    missing = sorted(set(expected) - set(df.columns))
    _check(not missing, f"R1 schema: {len(expected)} colunas esperadas presentes", report)

    num_ok = all(pd.api.types.is_numeric_dtype(df[c]) for c in cfg.NUMERIC_FEATURES)
    _check(num_ok, "R1 schema: colunas numéricas com dtype numérico", report)

    # R2 — alvo
    _check(df[cfg.TARGET].notna().all(), "R2 alvo: sem valores ausentes", report)
    levels = set(df[cfg.TARGET].unique())
    _check(levels == set(cfg.TARGET_MAP), f"R2 alvo: níveis == {set(cfg.TARGET_MAP)}", report)
    minority = df[cfg.TARGET].value_counts(normalize=True).min()
    _check(minority > 0.05, f"R2 alvo: classe minoritária = {minority:.1%} (> 5%)", report)

    # R3 — ausentes
    n_null = int(df.isna().sum().sum())
    _check(n_null == 0, f"R3 ausentes: {n_null} valores nulos no dataset", report)

    # R4 — duplicidade
    _check(df["Patient_ID"].is_unique, "R4 duplicidade: Patient_ID único", report)
    dup_rows = int(df.drop(columns=["Patient_ID"]).duplicated().sum())
    _check(dup_rows == 0, f"R4 duplicidade: {dup_rows} linhas idênticas (sem ID)", report)

    # R5 — domínio numérico
    for col, (lo, hi) in cfg.VALID_RANGES.items():
        out = int(((df[col] < lo) | (df[col] > hi)).sum())
        _check(out == 0, f"R5 domínio: {col} dentro de [{lo}, {hi}] ({out} fora)", report)

    # R6 — vocabulário categórico
    vocab = {
        "Gender": {"Male", "Female"},
        "Smoking_Status": {"Never Smoked", "Former Smoker", "Current Smoker"},
        "Cancer_Type": {"NSCLC", "SCLC"},
        "Cancer_Stage": set(cfg.STAGE_ORDER),
        "Metastasis": {"Yes", "No"},
    }
    for col, allowed in vocab.items():
        obs = set(df[col].unique())
        _check(obs <= allowed, f"R6 categorias: {col} ⊆ {sorted(allowed)}", report)

    for col in cfg.BINARY_FEATURES:
        obs = set(df[col].unique())
        _check(obs <= {"Yes", "No"}, f"R6 categorias: {col} binária Yes/No", report)

    # R7 — coerência clínica
    never = df["Smoking_Status"] == "Never Smoked"
    _check(
        bool((df.loc[never, ["Cigarettes_Per_Day", "Years_Smoking", "Pack_Years"]] == 0).all().all()),
        "R7 coerência: 'Never Smoked' => carga tabágica zero",
        report,
    )
    sclc = df["Cancer_Type"] == "SCLC"
    _check(
        bool((df.loc[sclc, "NSCLC_Subtype"] == "Not Applicable").all()),
        "R7 coerência: SCLC => NSCLC_Subtype == 'Not Applicable'",
        report,
    )
    _check(
        bool((df.loc[df["Cancer_Stage"] == "Stage IV", "Metastasis"] == "Yes").all()),
        "R7 coerência: Stage IV => Metastasis == 'Yes'",
        report,
    )

    # R9 — derivações exatas (documentadas em config.DERIVED_KEPT)
    pack = df["Cigarettes_Per_Day"] / 20.0 * df["Years_Smoking"]
    _check(
        bool(np.isclose(pack, df["Pack_Years"], atol=0.06).all()),
        "R9 derivação: Pack_Years == Cigarettes_Per_Day/20 * Years_Smoking",
        report,
    )
    symptoms = [
        "Coughing", "Shortness_of_Breath", "Chest_Pain", "Coughing_Blood",
        "Fatigue", "Weight_Loss", "Wheezing", "Recurrent_Infections",
        "Swallowing_Difficulty", "Finger_Clubbing",
    ]
    _check(
        bool(((df[symptoms] == "Yes").sum(axis=1) == df["Symptom_Count"]).all()),
        "R9 derivação: Symptom_Count == soma dos 10 sintomas",
        report,
    )
    risks = [
        "Secondhand_Smoke", "Family_History", "Occupational_Hazard",
        "Chronic_Lung_Disease", "Asbestos_Exposure", "Radon_Exposure",
        "Previous_Cancer_History",
    ]
    rsum = (df[risks] == "Yes").sum(axis=1) + (df["Air_Pollution_Exposure"] == "High").astype(int)
    _check(
        bool((rsum == df["Risk_Factor_Count"]).all()),
        "R9 derivação: Risk_Factor_Count == 7 fatores + poluição alta",
        report,
    )

    if verbose:
        print("\n".join(report))
    return report


# ---------------------------------------------------------------------------
# R8: validação anti-vazamento da matriz de features
# ---------------------------------------------------------------------------
def validate_no_leakage(X: pd.DataFrame, verbose: bool = True) -> list[str]:
    report = ["VALIDACAO ANTI-VAZAMENTO"]
    forbidden = set(cfg.DROP_ALWAYS) | {cfg.TARGET}
    present = sorted(forbidden & set(X.columns))
    _check(not present, f"R8 vazamento: nenhuma coluna proibida em X ({sorted(forbidden)})", report)
    if verbose:
        print("\n".join(report))
    return report


def validate_split(
    X_train: pd.DataFrame, X_test: pd.DataFrame, verbose: bool = True
) -> list[str]:
    """Garante que treino e teste são disjuntos (nenhum paciente nos dois lados)."""
    report = ["VALIDACAO DA DIVISAO TREINO/TESTE"]
    overlap = set(X_train.index) & set(X_test.index)
    _check(not overlap, f"divisão: 0 índices em comum ({len(overlap)} encontrados)", report)

    dup = pd.concat([X_train, X_test]).duplicated().sum()
    _check(
        int(dup) == 0,
        f"divisão: 0 linhas de features idênticas entre os conjuntos ({dup})",
        report,
    )
    if verbose:
        print("\n".join(report))
    return report

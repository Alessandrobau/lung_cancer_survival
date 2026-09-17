"""
ablation.py — Valor incremental de cada bloco de variáveis.

Executar: python -m src.ablation

A interpretação do modelo final mostrou que o estadiamento concentra quase todo
o poder preditivo. Este experimento quantifica isso: parte de um modelo com
apenas Cancer_Stage e adiciona blocos clínicos em ordem de disponibilidade,
medindo o ganho de ROC-AUC a cada passo com validação cruzada repetida (5x5),
o que permite comparar ganhos com o erro padrão da própria estimativa.

A pergunta que o experimento responde: as outras 30+ variáveis agregam sinal ou
o modelo é, na prática, um preditor de estádio?
"""

from __future__ import annotations

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from . import config as cfg
from . import data as data_mod
from . import evaluation as ev
from .features import build_preprocessor

warnings.filterwarnings("ignore")

RCV = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=cfg.RANDOM_STATE)

# Blocos em ordem de disponibilidade clínica no momento do diagnóstico
BLOCKS = [
    ("Estadiamento", ["Cancer_Stage"]),
    ("+ Tumor e metástase", ["Metastasis", "Tumor_Size_cm", "Cancer_Type", "NSCLC_Subtype"]),
    ("+ Método de diagnóstico", ["Diagnosis_Method"]),
    ("+ Demografia", ["Age", "Gender", "WHO_Region", "Diagnosis_Year"]),
    ("+ Tabagismo e estilo de vida",
     ["Smoking_Status", "Cigarettes_Per_Day", "Years_Smoking", "Pack_Years",
      "BMI", "Alcohol_Use", "Exercise_Frequency", "Air_Pollution_Exposure",
      "Secondhand_Smoke"]),
    ("+ Sintomas",
     ["Coughing", "Shortness_of_Breath", "Chest_Pain", "Coughing_Blood", "Fatigue",
      "Weight_Loss", "Wheezing", "Recurrent_Infections", "Swallowing_Difficulty",
      "Finger_Clubbing", "Symptom_Count"]),
    ("+ Genética e comorbidades",
     ["Genetic_Mutation", "Family_History", "Chronic_Lung_Disease",
      "Occupational_Hazard", "Asbestos_Exposure", "Radon_Exposure",
      "Previous_Cancer_History", "Risk_Factor_Count"]),
]


def _pipe_for(cols):
    """Pipeline restrito a um subconjunto de colunas."""
    import src.config as c

    num = [x for x in c.NUMERIC_FEATURES if x in cols]
    binr = [x for x in c.BINARY_FEATURES if x in cols]
    cat = [x for x in c.CATEGORICAL_FEATURES if x in cols]

    # build_preprocessor usa as listas globais; aqui monta-se uma versão local
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
    from .features import ORDINAL_SPEC

    ordc = [x for x in cat if x in ORDINAL_SPEC]
    nomc = [x for x in cat if x not in ORDINAL_SPEC]

    tr = []
    if num:
        tr.append(("num", Pipeline([("i", SimpleImputer(strategy="median")),
                                    ("s", StandardScaler())]), num))
    if binr:
        tr.append(("bin", Pipeline([
            ("i", SimpleImputer(strategy="most_frequent")),
            ("e", OrdinalEncoder(categories=[["No", "Yes"]] * len(binr),
                                 handle_unknown="use_encoded_value", unknown_value=-1)),
        ]), binr))
    if ordc:
        tr.append(("ord", Pipeline([
            ("i", SimpleImputer(strategy="most_frequent")),
            ("e", OrdinalEncoder(categories=[ORDINAL_SPEC[x] for x in ordc],
                                 handle_unknown="use_encoded_value", unknown_value=-1)),
            ("s", StandardScaler()),
        ]), ordc))
    if nomc:
        tr.append(("nom", Pipeline([
            ("i", SimpleImputer(strategy="most_frequent")),
            ("e", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                min_frequency=10, sparse_output=False)),
        ]), nomc))

    prep = ColumnTransformer(tr, remainder="drop", verbose_feature_names_out=False)
    clf = LogisticRegression(max_iter=5000, solver="liblinear", penalty="l1", C=0.1,
                             class_weight="balanced", random_state=cfg.RANDOM_STATE)
    return Pipeline([("prep", prep), ("clf", clf)])


def main():
    print("=" * 74)
    print("VALOR INCREMENTAL DE CADA BLOCO DE VARIAVEIS")
    print("=" * 74)
    df = data_mod.load_raw(verbose=False)
    X, y = data_mod.make_xy(df, verbose=False)
    X_tr, _, y_tr, _ = data_mod.split(X, y, verbose=False)

    print("Validação cruzada repetida 5x5 (25 estimativas por configuração).")
    print("Modelo fixo: Regressão Logística L1, C=0.1, class_weight=balanced.\n")

    rows, cum = [], []
    prev = None
    for name, cols in BLOCKS:
        cum += cols
        pipe = _pipe_for(cum)
        s = cross_val_score(pipe, X_tr, y_tr, cv=RCV, scoring="roc_auc", n_jobs=1)
        mean, se = float(s.mean()), float(s.std(ddof=1) / np.sqrt(len(s)))
        delta = (mean - prev) if prev is not None else np.nan
        rows.append({"bloco": name, "n_variaveis": len(cum), "roc_auc": mean,
                     "se": se, "ganho": delta})
        flag = ""
        if prev is not None:
            flag = "  <- ganho abaixo do erro padrão" if abs(delta) < se else ""
        print(f"  {name:<30} n={len(cum):>2}  ROC-AUC = {mean:.4f} +/- {se:.4f}"
              + (f"  ganho = {delta:+.4f}{flag}" if prev is not None else ""))
        prev = mean

    tbl = pd.DataFrame(rows)
    total = tbl["roc_auc"].iloc[-1] - tbl["roc_auc"].iloc[0]
    print(f"\n  Ganho total do bloco 2 ao 7 (33 variáveis): {total:+.4f} de ROC-AUC")
    print(f"  Erro padrão típico da estimativa           : {tbl['se'].mean():.4f}")
    print(f"  Razão ganho/erro padrão                    : {total/tbl['se'].mean():.2f}")

    # baseline absoluto: Cancer_Stage univariado, sem modelo
    from sklearn.metrics import roc_auc_score
    stage_num = df["Cancer_Stage"].map({s: i for i, s in enumerate(cfg.STAGE_ORDER)})
    auc_uni = roc_auc_score((df[cfg.TARGET] == "Yes").astype(int), -stage_num)
    print(f"\n  Referência: ordenar pacientes só pelo estádio -> ROC-AUC = {auc_uni:.4f}")

    # ---- figura
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))
    x = np.arange(len(tbl))
    axes[0].errorbar(x, tbl["roc_auc"], yerr=tbl["se"], fmt="o-", color="#2471a3",
                     lw=1.8, capsize=4, ms=6)
    axes[0].axhline(auc_uni, ls="--", color="#c0392b", lw=1.2,
                    label=f"só o estádio, sem modelo = {auc_uni:.3f}")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([b.replace("+ ", "+\n") for b in tbl["bloco"]],
                            rotation=0, fontsize=7)
    axes[0].set_ylabel("ROC-AUC (CV repetida 5x5)")
    axes[0].set_title("Desempenho acumulado ao adicionar blocos de variáveis", fontsize=9)
    axes[0].legend(fontsize=8, frameon=False)

    g = tbl["ganho"].iloc[1:]
    colors = ["#2e7d32" if abs(v) > tbl["se"].mean() else "#95a5a6" for v in g]
    axes[1].bar(np.arange(1, len(tbl)), g, color=colors, alpha=0.9)
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].axhline(tbl["se"].mean(), ls=":", color="#c0392b", lw=1.2,
                    label="erro padrão da estimativa")
    axes[1].axhline(-tbl["se"].mean(), ls=":", color="#c0392b", lw=1.2)
    axes[1].set_xticks(np.arange(1, len(tbl)))
    axes[1].set_xticklabels([b.replace("+ ", "+\n") for b in tbl["bloco"].iloc[1:]],
                            fontsize=7)
    axes[1].set_ylabel("Ganho de ROC-AUC sobre o bloco anterior")
    axes[1].set_title("Ganho marginal por bloco\n(cinza = dentro do ruído da estimativa)",
                      fontsize=9)
    axes[1].legend(fontsize=8, frameon=False)

    for ax in axes:
        ax.grid(alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out = cfg.FIG_DIR / "14_ablacao_blocos.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigura -> {out.relative_to(cfg.ROOT)}")

    tbl.to_csv(cfg.ARTIFACTS_DIR / "ablacao_blocos.csv", index=False)
    ev.save_json({"blocos": tbl.to_dict("records"), "auc_estadio_univariado": float(auc_uni)},
                 cfg.ARTIFACTS_DIR / "ablacao_blocos.json")


if __name__ == "__main__":
    main()

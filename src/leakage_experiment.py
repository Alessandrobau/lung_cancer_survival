"""
leakage_experiment.py — Diagnóstico e quantificação do vazamento de dados.

Executar: python -m src.leakage_experiment

Três partes:

  PARTE 1 — Ablação das variáveis pós-desfecho
      Mede quanto Treatment e Survival_Months mudam o desempenho aparente.
      Cada configuração tem seus próprios hiperparâmetros ajustados por CV no
      treino, para que a comparação seja justa.

  PARTE 2 — Por que a ablação dá no que dá
      Mostra a redundância entre as variáveis vazadas e o estadiamento, que já
      está disponível no momento do diagnóstico.

  PARTE 3 — Vazamento de pré-processamento (demonstração controlada)
      Rótulos aleatórios e atributos de puro ruído. O desempenho real é 0,50 por
      construção. Ajustar a seleção de atributos no dataset inteiro produz um
      AUC aparente muito acima disso; fazer a mesma seleção dentro do Pipeline
      devolve o resultado correto. É exatamente o erro que a arquitetura deste
      projeto impede por construção.
"""

from __future__ import annotations

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from . import config as cfg
from . import data as data_mod
from . import evaluation as ev
from .features import build_preprocessor

warnings.filterwarnings("ignore")

CV = StratifiedKFold(n_splits=cfg.N_SPLITS, shuffle=True, random_state=cfg.RANDOM_STATE)
GRID = {"clf__C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0], "clf__penalty": ["l1", "l2"]}


# ---------------------------------------------------------------------------
# PARTE 1 — ablação
# ---------------------------------------------------------------------------
def run_config(name, X, y, extra_numeric=None, extra_nominal=None):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=cfg.TEST_SIZE, random_state=cfg.RANDOM_STATE, stratify=y
    )
    pipe = Pipeline(
        [
            ("prep", build_preprocessor(True, extra_numeric, extra_nominal)),
            (
                "clf",
                LogisticRegression(
                    max_iter=5000, solver="liblinear",
                    class_weight="balanced", random_state=cfg.RANDOM_STATE,
                ),
            ),
        ]
    )
    gs = GridSearchCV(pipe, GRID, cv=CV, scoring="roc_auc", n_jobs=1).fit(X_tr, y_tr)
    p_te = gs.predict_proba(X_te)[:, 1]
    return {
        "config": name,
        "cv_roc_auc": float(gs.best_score_),
        "test_roc_auc": float(roc_auc_score(y_te, p_te)),
        "best_params": str({k.replace("clf__", ""): v for k, v in gs.best_params_.items()}),
    }, y_te, p_te


def parte1(df):
    print("\n" + "=" * 74)
    print("PARTE 1 — ABLACAO DAS VARIAVEIS POS-DESFECHO")
    print("=" * 74)
    X_clean, y = data_mod.make_xy(df, include_leaky=False, verbose=False)
    X_leaky, _ = data_mod.make_xy(df, include_leaky=True, verbose=False)

    results, curves = [], {}
    specs = [
        ("A. Sem vazamento (modelo entregue)", X_clean, None, None),
        ("B. + Treatment", X_leaky.drop(columns=["Survival_Months"]), None, ["Treatment"]),
        ("C. + Survival_Months", X_leaky.drop(columns=["Treatment"]), ["Survival_Months"], None),
        ("D. + ambas", X_leaky, ["Survival_Months"], ["Treatment"]),
    ]
    for name, X, en, eo in specs:
        r, yt, pt = run_config(name, X, y, en, eo)
        results.append(r)
        curves[name] = (yt, pt)
        print(f"  {name:<36} CV = {r['cv_roc_auc']:.4f} | teste = {r['test_roc_auc']:.4f} "
              f"| {r['best_params']}")

    tbl = pd.DataFrame(results).set_index("config")
    base = tbl.loc["A. Sem vazamento (modelo entregue)", "test_roc_auc"]
    tbl["delta_vs_A"] = tbl["test_roc_auc"] - base

    print("\n  Variação do ROC-AUC de teste em relação ao modelo sem vazamento:")
    for idx, row in tbl.iterrows():
        if idx.startswith("A."):
            continue
        print(f"    {idx:<34} {row['delta_vs_A']:+.4f}")
    print(
        "\n  Leitura: neste dataset as variáveis pós-desfecho NAO inflam o\n"
        "  desempenho — são redundantes com o estadiamento (ver Parte 2).\n"
        "  A remoção continua obrigatória por razão metodológica: no uso real\n"
        "  elas não existem no instante em que a predição precisa ser feita."
    )
    return tbl, curves


# ---------------------------------------------------------------------------
# PARTE 2 — por que
# ---------------------------------------------------------------------------
def parte2(df):
    print("\n" + "=" * 74)
    print("PARTE 2 — REDUNDANCIA ENTRE AS VARIAVEIS VAZADAS E O ESTADIAMENTO")
    print("=" * 74)
    y = (df[cfg.TARGET] == "Yes").astype(int)
    stage_num = df["Cancer_Stage"].map({s: i for i, s in enumerate(cfg.STAGE_ORDER)})

    auc_sm = roc_auc_score(y, df["Survival_Months"])
    auc_st = roc_auc_score(y, -stage_num)
    rho = float(df["Survival_Months"].corr(stage_num, method="spearman"))

    print(f"  AUC univariada de Survival_Months          : {auc_sm:.4f}")
    print(f"  AUC univariada de Cancer_Stage             : {auc_st:.4f}")
    print(f"  Correlação Spearman Survival_Months x Stage: {rho:.4f}")

    print("\n  Mediana de Survival_Months por estádio:")
    med = df.groupby("Cancer_Stage")["Survival_Months"].median()
    for s in cfg.STAGE_ORDER:
        print(f"    {s:<10} {med[s]:>6.1f} meses")

    print("\n  Taxa de sobrevida por tratamento (Treatment já reflete o prognóstico):")
    rate = (df.groupby("Treatment")[cfg.TARGET]
              .apply(lambda s: (s == "Yes").mean() * 100).sort_values())
    for k, v in rate.items():
        print(f"    {k:<24} {v:>5.1f}%")

    # incoerência temporal: prova de que Survival_Months não é observável
    fu_max = 12 * (2026 - df["Diagnosis_Year"]) + 12
    n_imp = int((df["Survival_Months"] > fu_max).sum())
    print(f"\n  Casos com sobrevida maior que o seguimento máximo possível: "
          f"{n_imp} ({100*n_imp/len(df):.1f}%)")
    print("  -> Survival_Months é um valor atribuído após o desfecho, não uma")
    print("     medição disponível na data do diagnóstico.")

    return {"auc_survival_months": float(auc_sm), "auc_cancer_stage": float(auc_st),
            "spearman_surv_stage": rho, "casos_temporalmente_impossiveis": n_imp}


# ---------------------------------------------------------------------------
# PARTE 3 — vazamento de pré-processamento
# ---------------------------------------------------------------------------
def parte3(n=2000, p=3000, k=20, seed=cfg.RANDOM_STATE):
    print("\n" + "=" * 74)
    print("PARTE 3 — VAZAMENTO DE PRE-PROCESSAMENTO (DEMONSTRACAO CONTROLADA)")
    print("=" * 74)
    rng = np.random.default_rng(seed)
    Z = pd.DataFrame(rng.normal(size=(n, p)))
    y = pd.Series(rng.integers(0, 2, n))
    print(f"  {n} amostras, {p} atributos de ruído puro, rótulos aleatórios.")
    print(f"  Desempenho verdadeiro por construção: ROC-AUC = 0.500\n")

    # ERRADO — seleção ajustada no dataset inteiro, antes do split
    sel = SelectKBest(f_classif, k=k).fit(Z, y)
    Z_sel = sel.transform(Z)
    Ztr, Zte, ytr, yte = train_test_split(
        Z_sel, y, test_size=cfg.TEST_SIZE, random_state=seed, stratify=y
    )
    clf = LogisticRegression(max_iter=3000).fit(Ztr, ytr)
    p_wrong = clf.predict_proba(Zte)[:, 1]
    auc_wrong = roc_auc_score(yte, p_wrong)

    # CERTO — seleção dentro do Pipeline, refeita a cada ajuste
    Ztr, Zte, ytr, yte = train_test_split(
        Z, y, test_size=cfg.TEST_SIZE, random_state=seed, stratify=y
    )
    pipe = Pipeline(
        [("sel", SelectKBest(f_classif, k=k)), ("clf", LogisticRegression(max_iter=3000))]
    ).fit(Ztr, ytr)
    p_right = pipe.predict_proba(Zte)[:, 1]
    auc_right = roc_auc_score(yte, p_right)

    print(f"  [ERRADO] seleção de {k} atributos no dataset inteiro -> AUC = {auc_wrong:.4f}")
    print(f"  [CERTO ] seleção dentro do Pipeline                  -> AUC = {auc_right:.4f}")
    print(f"\n  Ilusão de desempenho criada pelo vazamento: {auc_wrong - 0.5:+.4f} de AUC")
    print("  sobre dados que, por construção, não contêm sinal algum.")

    return {"auc_wrong": float(auc_wrong), "auc_right": float(auc_right),
            "n": n, "p": p, "k": k}, (yte, p_wrong, p_right)


# ---------------------------------------------------------------------------
def figura(tbl, curves, demo):
    yte, p_wrong, p_right = demo
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    colors = ["#2e7d32", "#f39c12", "#c0392b", "#8e44ad"]

    for (name, (yt, pt)), c in zip(curves.items(), colors):
        fpr, tpr, _ = roc_curve(yt, pt)
        axes[0].plot(fpr, tpr, color=c, lw=1.8,
                     label=f"{name.split('.')[0]}. AUC={tbl.loc[name,'test_roc_auc']:.3f}")
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[0].set_title("Parte 1 — ROC por configuração\n(teste hold-out)", fontsize=9)
    axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
    axes[0].legend(fontsize=7.5, loc="lower right", frameon=False)

    labels = [n.split(".")[0] for n in tbl.index]
    axes[1].bar(labels, tbl["test_roc_auc"], color=colors, alpha=0.9)
    axes[1].axhline(tbl["test_roc_auc"].iloc[0], ls="--", color="black", lw=1)
    for i, v in enumerate(tbl["test_roc_auc"]):
        axes[1].text(i, v + 0.004, f"{v:.3f}", ha="center", fontsize=8.5)
    axes[1].set_ylim(0.70, 0.83)
    axes[1].set_ylabel("ROC-AUC no teste")
    axes[1].set_title("Parte 1 — variáveis pós-desfecho\nnão inflam o resultado neste dataset",
                      fontsize=9)

    fpr, tpr, _ = roc_curve(yte, p_wrong)
    axes[2].plot(fpr, tpr, color="#c0392b", lw=2,
                 label=f"seleção no dataset todo\nAUC={roc_auc_score(yte,p_wrong):.3f}")
    fpr, tpr, _ = roc_curve(yte, p_right)
    axes[2].plot(fpr, tpr, color="#2e7d32", lw=2,
                 label=f"seleção no Pipeline\nAUC={roc_auc_score(yte,p_right):.3f}")
    axes[2].plot([0, 1], [0, 1], "k--", lw=0.8, label="verdade = 0.50")
    axes[2].set_title("Parte 3 — ruído puro, rótulos aleatórios:\nvazamento de pré-processamento",
                      fontsize=9)
    axes[2].set_xlabel("FPR"); axes[2].set_ylabel("TPR")
    axes[2].legend(fontsize=7, loc="lower right", frameon=False)

    for ax in axes:
        ax.grid(alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out = cfg.FIG_DIR / "07_experimento_vazamento.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigura -> {out.relative_to(cfg.ROOT)}")


def main():
    print("=" * 74)
    print("EXPERIMENTO — DIAGNOSTICO DE VAZAMENTO DE DADOS")
    print("=" * 74)
    df = data_mod.load_raw(verbose=False)

    tbl, curves = parte1(df)
    p2 = parte2(df)
    p3, demo = parte3()
    figura(tbl, curves, demo)

    tbl.to_csv(cfg.ARTIFACTS_DIR / "leakage_experiment.csv")
    ev.save_json(
        {"ablacao": tbl.to_dict("index"), "redundancia": p2, "preproc_demo": p3},
        cfg.ARTIFACTS_DIR / "leakage_experiment.json",
    )


if __name__ == "__main__":
    main()

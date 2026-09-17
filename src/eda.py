"""
eda.py — Análise exploratória dos dados.

Executar: python -m src.eda
Gera as figuras em reports/figures/ e um resumo em artifacts/eda_summary.json.
"""

from __future__ import annotations

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as cfg
from . import data as data_mod
from .evaluation import save_json

warnings.filterwarnings("ignore")

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 130,
        "font.size": 9,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
    }
)

C_NEG, C_POS = "#c0392b", "#2471a3"


def _save(fig, name):
    path = cfg.FIG_DIR / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  figura -> {path.relative_to(cfg.ROOT)}")
    return path


# ---------------------------------------------------------------------------
def fig_target_and_missing(df):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))

    vc = df[cfg.TARGET].value_counts()
    axes[0].bar(["Não sobreviveu", "Sobreviveu"], [vc["No"], vc["Yes"]],
                color=[C_NEG, C_POS])
    for i, v in enumerate([vc["No"], vc["Yes"]]):
        axes[0].text(i, v + 15, f"{v}\n({v/len(df):.1%})", ha="center", fontsize=8)
    axes[0].set_title("Distribuição do alvo (Survived)")
    axes[0].set_ylim(0, vc.max() * 1.25)
    axes[0].set_ylabel("pacientes")

    miss = df.isna().mean().sort_values(ascending=False).head(10)
    axes[1].barh(range(len(miss)), miss.values * 100, color="#7f8c8d")
    axes[1].set_yticks(range(len(miss)))
    axes[1].set_yticklabels(miss.index, fontsize=7)
    axes[1].set_xlim(0, 100)
    axes[1].set_title("Valores ausentes (%) — top 10")
    axes[1].text(50, len(miss) / 2, "0% de ausentes\nem todas as colunas",
                 ha="center", va="center", fontsize=9, color="#2e7d32", weight="bold")

    yr = df.groupby("Diagnosis_Year")[cfg.TARGET].apply(lambda s: (s == "Yes").mean() * 100)
    axes[2].plot(yr.index, yr.values, marker="o", color=C_POS)
    axes[2].axhline(37.8, ls="--", color="grey", lw=1)
    axes[2].set_title("Taxa de sobrevida por ano de diagnóstico")
    axes[2].set_xlabel("ano")
    axes[2].set_ylabel("% sobreviventes")
    axes[2].set_ylim(0, 60)

    return _save(fig, "01_eda_alvo.png")


def fig_numeric_distributions(df):
    cols = ["Age", "BMI", "Tumor_Size_cm", "Pack_Years", "Symptom_Count", "Risk_Factor_Count"]
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.4))
    for ax, c in zip(axes.ravel(), cols):
        for lab, color, name in [("No", C_NEG, "não sobreviveu"), ("Yes", C_POS, "sobreviveu")]:
            ax.hist(df.loc[df[cfg.TARGET] == lab, c], bins=25, alpha=0.55,
                    color=color, label=name, density=True)
        ax.set_title(c, fontsize=9)
        ax.set_yticks([])
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.suptitle("Distribuição das variáveis numéricas por classe", y=1.0, fontsize=10)
    return _save(fig, "02_eda_numericas.png")


def fig_outliers(df):
    cols = ["Age", "BMI", "Tumor_Size_cm", "Pack_Years", "Cigarettes_Per_Day", "Years_Smoking"]
    fig, axes = plt.subplots(1, 6, figsize=(11, 3.0))
    rows = []
    for ax, c in zip(axes, cols):
        q1, q3 = df[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = int(((df[c] < lo) | (df[c] > hi)).sum())
        rows.append({"variavel": c, "outliers_IQR": n_out, "pct": round(100 * n_out / len(df), 2),
                     "min": float(df[c].min()), "max": float(df[c].max())})
        ax.boxplot(df[c], widths=0.5, patch_artist=True,
                   boxprops=dict(facecolor="#d6eaf8", color="#2471a3"),
                   medianprops=dict(color="#c0392b"))
        ax.set_title(f"{c}\n{n_out} out. ({100*n_out/len(df):.1f}%)", fontsize=7.5)
        ax.set_xticks([])
    fig.suptitle("Boxplots e contagem de outliers pelo critério IQR (1,5×)", y=1.02, fontsize=10)
    _save(fig, "03_eda_outliers.png")
    return pd.DataFrame(rows)


def fig_categorical_rates(df):
    cols = ["Cancer_Stage", "Metastasis", "Diagnosis_Method", "Smoking_Status",
            "Cancer_Type", "Chronic_Lung_Disease", "Gender", "WHO_Region"]
    fig, axes = plt.subplots(2, 4, figsize=(12.5, 5.6))
    base = (df[cfg.TARGET] == "Yes").mean() * 100
    for ax, c in zip(axes.ravel(), cols):
        rate = (
            df.groupby(c)[cfg.TARGET].apply(lambda s: (s == "Yes").mean() * 100).sort_values()
        )
        colors = [C_POS if v > base else C_NEG for v in rate.values]
        ax.barh(range(len(rate)), rate.values, color=colors, alpha=0.85)
        ax.axvline(base, ls="--", lw=1, color="black")
        ax.set_yticks(range(len(rate)))
        ax.set_yticklabels(rate.index, fontsize=7)
        ax.set_xlim(0, 100)
        ax.set_title(c, fontsize=8.5)
    fig.suptitle(
        f"Taxa de sobrevida (%) por categoria — linha tracejada = taxa global ({base:.1f}%)",
        y=1.01, fontsize=10,
    )
    return _save(fig, "04_eda_categoricas.png")


def fig_correlation(df):
    num = df[cfg.NUMERIC_FEATURES].copy()
    num["Survived"] = (df[cfg.TARGET] == "Yes").astype(int)
    num["Survival_Months (VAZAMENTO)"] = df["Survival_Months"]
    corr = num.corr(method="spearman")

    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(corr)))
    ax.set_yticklabels(corr.columns, fontsize=7)
    for i in range(len(corr)):
        for j in range(len(corr)):
            v = corr.iloc[i, j]
            if abs(v) > 0.25:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if abs(v) > 0.6 else "black")
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Spearman ρ")
    ax.set_title("Correlação de Spearman — numéricas + alvo", fontsize=10)
    return _save(fig, "05_eda_correlacao.png")


def fig_leakage_evidence(df):
    """Evidência gráfica de que Survival_Months e Treatment vazam o desfecho."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))

    for lab, color, name in [("No", C_NEG, "não sobreviveu"), ("Yes", C_POS, "sobreviveu")]:
        axes[0].hist(df.loc[df[cfg.TARGET] == lab, "Survival_Months"], bins=30,
                     alpha=0.6, color=color, label=name, density=True)
    axes[0].set_title("Survival_Months por classe\n(variável pós-desfecho)", fontsize=9)
    axes[0].set_xlabel("meses"); axes[0].set_yticks([]); axes[0].legend(fontsize=7, frameon=False)

    # incoerência temporal: seguimento longo demais para diagnósticos recentes
    fu = 12 * (2026 - df["Diagnosis_Year"]) + 12
    impossible = df["Survival_Months"] > fu
    axes[1].scatter(df["Diagnosis_Year"] + np.random.uniform(-.25, .25, len(df)),
                    df["Survival_Months"], s=4, alpha=0.30,
                    c=np.where(impossible, "#c0392b", "#95a5a6"))
    axes[1].plot(sorted(df["Diagnosis_Year"].unique()),
                 [12 * (2026 - y) + 12 for y in sorted(df["Diagnosis_Year"].unique())],
                 color="black", lw=1.4, label="seguimento máximo possível")
    axes[1].set_title(f"Sobrevida vs. ano do diagnóstico\n{impossible.sum()} casos impossíveis (vermelho)",
                      fontsize=9)
    axes[1].set_xlabel("ano do diagnóstico"); axes[1].set_ylabel("Survival_Months")
    axes[1].legend(fontsize=7, frameon=False)

    rate = (df.groupby("Treatment")[cfg.TARGET]
              .apply(lambda s: (s == "Yes").mean() * 100).sort_values())
    axes[2].barh(range(len(rate)), rate.values, color="#8e44ad", alpha=0.85)
    axes[2].set_yticks(range(len(rate)))
    axes[2].set_yticklabels(rate.index, fontsize=7)
    axes[2].axvline(37.8, ls="--", lw=1, color="black")
    axes[2].set_title("Treatment — decisão posterior\nao instante de predição", fontsize=9)
    axes[2].set_xlabel("% sobrevida")

    return _save(fig, "06_eda_vazamento.png")


# ---------------------------------------------------------------------------
def main():
    print("=" * 74)
    print("ANALISE EXPLORATORIA DOS DADOS (EDA)")
    print("=" * 74)
    df = data_mod.load_raw()

    print(f"\nDimensões: {df.shape[0]} linhas x {df.shape[1]} colunas")
    print(f"Ausentes : {int(df.isna().sum().sum())}")
    print(f"Duplicados: {int(df.drop(columns=['Patient_ID']).duplicated().sum())}")
    print(f"Alvo     : {df[cfg.TARGET].value_counts().to_dict()}  "
          f"(razão {(df[cfg.TARGET]=='No').sum()/(df[cfg.TARGET]=='Yes').sum():.2f}:1)")

    print("\nGerando figuras...")
    fig_target_and_missing(df)
    fig_numeric_distributions(df)
    out_tbl = fig_outliers(df)
    fig_categorical_rates(df)
    fig_correlation(df)
    fig_leakage_evidence(df)

    print("\nOutliers (critério IQR):")
    print(out_tbl.to_string(index=False))

    # associação categórica x alvo (Cramér's V)
    from scipy.stats import chi2_contingency
    assoc = {}
    for c in cfg.CATEGORICAL_FEATURES + cfg.BINARY_FEATURES:
        tab = pd.crosstab(df[c], df[cfg.TARGET])
        chi2 = chi2_contingency(tab)[0]
        n = tab.values.sum()
        v = np.sqrt(chi2 / (n * (min(tab.shape) - 1)))
        assoc[c] = round(float(v), 4)
    assoc = dict(sorted(assoc.items(), key=lambda kv: -kv[1]))
    print("\nAssociação com o alvo (Cramér's V) — top 10:")
    for k, v in list(assoc.items())[:10]:
        print(f"  {k:<26} {v:.3f}")

    summary = {
        "n_linhas": int(df.shape[0]),
        "n_colunas": int(df.shape[1]),
        "ausentes": int(df.isna().sum().sum()),
        "duplicados": int(df.drop(columns=["Patient_ID"]).duplicated().sum()),
        "alvo": df[cfg.TARGET].value_counts().to_dict(),
        "outliers_iqr": out_tbl.to_dict("records"),
        "cramers_v": assoc,
    }
    save_json(summary, cfg.ARTIFACTS_DIR / "eda_summary.json")
    print(f"\nResumo salvo em artifacts/eda_summary.json")


if __name__ == "__main__":
    main()

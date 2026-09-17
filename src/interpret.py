"""
interpret.py — Avaliação visual e interpretação do modelo final.

Executar: python -m src.interpret   (requer models/model.joblib)

Gera:
  08_avaliacao.png        matriz de confusão, ROC, precisão-revocação, calibração
  09_limiar.png           métricas em função do limiar + distribuição de escores
  10_curva_aprendizado.png diagnóstico de underfitting / overfitting
  11_importancia.png      importância por permutação + coeficientes da regressão
  12_shap.png             beeswarm SHAP (efeito global e direção)
  13_shap_casos.png       explicação SHAP de casos individuais
"""

from __future__ import annotations

import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, learning_curve

from . import config as cfg
from . import data as data_mod
from . import evaluation as ev

warnings.filterwarnings("ignore")

plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
                     "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white"})

C_NEG, C_POS = "#c0392b", "#2471a3"


def _save(fig, name):
    path = cfg.FIG_DIR / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  figura -> {path.relative_to(cfg.ROOT)}")


# ---------------------------------------------------------------------------
def fig_evaluation(y_te, p_te, thr):
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))

    y_pred = (p_te >= thr).astype(int)
    cm = confusion_matrix(y_te, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=["Não sobrev.", "Sobreviveu"]).plot(
        ax=axes[0], cmap="Blues", colorbar=False, values_format="d"
    )
    axes[0].set_title(f"Matriz de confusão (limiar {thr:.2f})", fontsize=9)
    axes[0].set_xlabel("Classe prevista"); axes[0].set_ylabel("Classe real")
    axes[0].grid(False)

    fpr, tpr, _ = roc_curve(y_te, p_te)
    axes[1].plot(fpr, tpr, color=C_POS, lw=2, label=f"AUC = {roc_auc_score(y_te,p_te):.3f}")
    axes[1].plot([0, 1], [0, 1], "k--", lw=0.8, label="aleatório = 0.500")
    axes[1].set_xlabel("Taxa de falsos positivos"); axes[1].set_ylabel("Taxa de verdadeiros positivos")
    axes[1].set_title("Curva ROC — teste", fontsize=9)
    axes[1].legend(fontsize=8, loc="lower right", frameon=False)

    prec, rec, _ = precision_recall_curve(y_te, p_te)
    axes[2].plot(rec, prec, color="#8e44ad", lw=2,
                 label=f"AP = {average_precision_score(y_te,p_te):.3f}")
    axes[2].axhline(y_te.mean(), ls="--", color="k", lw=0.8,
                    label=f"base = {y_te.mean():.3f}")
    axes[2].set_xlabel("Revocação"); axes[2].set_ylabel("Precisão")
    axes[2].set_title("Curva precisão-revocação", fontsize=9)
    axes[2].legend(fontsize=8, loc="lower left", frameon=False)

    frac, mean_pred = calibration_curve(y_te, p_te, n_bins=8, strategy="quantile")
    axes[3].plot(mean_pred, frac, "o-", color="#e67e22", lw=1.8, label="modelo")
    axes[3].plot([0, 1], [0, 1], "k--", lw=0.8, label="calibração perfeita")
    axes[3].set_xlabel("Probabilidade prevista"); axes[3].set_ylabel("Frequência observada")
    axes[3].set_title(f"Calibração (Brier = {ev.compute_metrics(y_te,p_te,thr)['brier']:.3f})",
                      fontsize=9)
    axes[3].legend(fontsize=8, loc="upper left", frameon=False)

    _save(fig, "08_avaliacao.png")


def fig_threshold(y_tr, oof, y_te, p_te, thr):
    from sklearn.metrics import f1_score, precision_score, recall_score
    grid = np.linspace(0.05, 0.95, 181)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))

    for name, yy, pp, ls in [("treino (OOF)", y_tr, oof, "-"), ("teste", y_te, p_te, "--")]:
        f1 = [f1_score(yy, pp >= t, zero_division=0) for t in grid]
        axes[0].plot(grid, f1, ls, color=C_POS, lw=1.8, label=f"F1 — {name}")
    prec = [precision_score(y_te, p_te >= t, zero_division=0) for t in grid]
    rec = [recall_score(y_te, p_te >= t, zero_division=0) for t in grid]
    axes[0].plot(grid, prec, color="#8e44ad", lw=1.2, alpha=0.8, label="precisão — teste")
    axes[0].plot(grid, rec, color="#e67e22", lw=1.2, alpha=0.8, label="revocação — teste")
    axes[0].axvline(thr, color="black", ls=":", lw=1.5, label=f"limiar escolhido = {thr:.3f}")
    axes[0].axvline(0.5, color="grey", ls=":", lw=1)
    axes[0].set_xlabel("Limiar de decisão"); axes[0].set_ylabel("Métrica")
    axes[0].set_title("Métricas em função do limiar\n(limiar calibrado no treino, via CV)",
                      fontsize=9)
    axes[0].legend(fontsize=7.5, frameon=False)

    axes[1].hist(p_te[y_te == 0], bins=28, alpha=0.6, color=C_NEG,
                 label="não sobreviveu", density=True)
    axes[1].hist(p_te[y_te == 1], bins=28, alpha=0.6, color=C_POS,
                 label="sobreviveu", density=True)
    axes[1].axvline(thr, color="black", ls=":", lw=1.5)
    axes[1].set_xlabel("Probabilidade prevista de sobrevida")
    axes[1].set_yticks([])
    axes[1].set_title("Separação das classes no teste", fontsize=9)
    axes[1].legend(fontsize=8, frameon=False)

    _save(fig, "09_limiar.png")


def fig_learning_curve(model, X_tr, y_tr):
    cv = StratifiedKFold(cfg.N_SPLITS, shuffle=True, random_state=cfg.RANDOM_STATE)
    sizes, tr, te = learning_curve(
        model, X_tr, y_tr, cv=cv, scoring="roc_auc",
        train_sizes=np.linspace(0.1, 1.0, 8), n_jobs=1, random_state=cfg.RANDOM_STATE,
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(sizes, tr.mean(1), "o-", color=C_NEG, label="treino")
    ax.fill_between(sizes, tr.mean(1) - tr.std(1), tr.mean(1) + tr.std(1),
                    color=C_NEG, alpha=0.15)
    ax.plot(sizes, te.mean(1), "o-", color=C_POS, label="validação (CV)")
    ax.fill_between(sizes, te.mean(1) - te.std(1), te.mean(1) + te.std(1),
                    color=C_POS, alpha=0.15)
    ax.set_xlabel("Amostras de treino"); ax.set_ylabel("ROC-AUC")
    ax.set_title(f"Curva de aprendizado\ngap final = {tr.mean(1)[-1]-te.mean(1)[-1]:+.4f}",
                 fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    _save(fig, "10_curva_aprendizado.png")
    return {"train_sizes": sizes.tolist(), "train": tr.mean(1).tolist(),
            "cv": te.mean(1).tolist()}


def fig_importance(model, X_te, y_te, top=18):
    r = permutation_importance(
        model, X_te, y_te, n_repeats=20, random_state=cfg.RANDOM_STATE,
        scoring="roc_auc", n_jobs=1,
    )
    imp = (pd.DataFrame({"feature": X_te.columns, "mean": r.importances_mean,
                         "std": r.importances_std})
           .sort_values("mean", ascending=False).head(top).iloc[::-1])

    prep = model.named_steps["prep"]
    clf = model.named_steps["clf"]
    names = list(prep.get_feature_names_out())
    coefs = pd.Series(clf.coef_[0], index=names)
    nz = coefs[coefs != 0].sort_values()
    sel = pd.concat([nz.head(9), nz.tail(9)]).drop_duplicates()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    axes[0].barh(imp["feature"], imp["mean"], xerr=imp["std"],
                 color="#2471a3", alpha=0.85, error_kw=dict(lw=0.8, alpha=0.5))
    axes[0].set_xlabel("Queda média de ROC-AUC ao embaralhar a variável")
    axes[0].set_title(f"Importância por permutação — teste (top {top})", fontsize=9)
    axes[0].tick_params(axis="y", labelsize=7.5)

    colors = [C_POS if v > 0 else C_NEG for v in sel.values]
    axes[1].barh(sel.index, sel.values, color=colors, alpha=0.85)
    axes[1].axvline(0, color="black", lw=0.8)
    axes[1].set_xlabel("Coeficiente (log-odds, variáveis padronizadas)")
    axes[1].set_title(f"Coeficientes da Regressão Logística\n"
                      f"{(coefs!=0).sum()} de {len(coefs)} não nulos (L1)", fontsize=9)
    axes[1].tick_params(axis="y", labelsize=7.5)

    _save(fig, "11_importancia.png")
    return imp.iloc[::-1], coefs


def fig_shap(model, X_tr, X_te):
    try:
        import shap
    except ImportError:
        print("  SHAP indisponível — etapa ignorada.")
        return None

    prep = model.named_steps["prep"]
    clf = model.named_steps["clf"]
    names = list(prep.get_feature_names_out())
    Z_tr = prep.transform(X_tr)
    Z_te = prep.transform(X_te)

    # Modelo linear -> explainer exato, sem amostragem
    explainer = shap.LinearExplainer(clf, Z_tr, feature_names=names)
    sv = explainer(Z_te)

    fig = plt.figure(figsize=(8.5, 6))
    shap.plots.beeswarm(sv, max_display=18, show=False)
    plt.title("SHAP — contribuição de cada variável para a log-odds de sobrevida",
              fontsize=10)
    _save(fig, "12_shap.png")

    # Casos individuais: um sobrevivente provável, um não sobrevivente provável
    p = clf.predict_proba(Z_te)[:, 1]
    idx_hi, idx_lo = int(np.argmax(p)), int(np.argmin(p))
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 7.5))
    for ax, idx, label in [
        (axes[0], idx_hi, f"Caso {X_te.index[idx_hi]} — prob. prevista {p[idx_hi]:.3f}"),
        (axes[1], idx_lo, f"Caso {X_te.index[idx_lo]} — prob. prevista {p[idx_lo]:.3f}"),
    ]:
        contrib = pd.Series(sv.values[idx], index=names)
        top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(12).iloc[::-1]
        ax.barh(top.index, top.values,
                color=[C_POS if v > 0 else C_NEG for v in top.values], alpha=0.85)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("contribuição SHAP (log-odds)")
        ax.tick_params(axis="y", labelsize=7.5)
    _save(fig, "13_shap_casos.png")

    mean_abs = pd.Series(np.abs(sv.values).mean(0), index=names).sort_values(ascending=False)
    return mean_abs


# ---------------------------------------------------------------------------
def main():
    print("=" * 74)
    print("AVALIACAO VISUAL E INTERPRETACAO DO MODELO FINAL")
    print("=" * 74)

    bundle = joblib.load(cfg.MODELS_DIR / "model.joblib")
    model, thr = bundle["model"], bundle["threshold"]
    print(f"Modelo: {bundle['model_name']}  |  limiar = {thr:.3f}")
    print(f"Hiperparâmetros: {bundle['best_params']}")

    df = data_mod.load_raw(verbose=False)
    X, y = data_mod.make_xy(df, verbose=False)
    X_tr, X_te, y_tr, y_te = data_mod.split(X, y, verbose=False)

    p_te = model.predict_proba(X_te)[:, 1]
    oof = np.load(cfg.ARTIFACTS_DIR / "proba_oof.npy")

    print("\nGerando figuras...")
    fig_evaluation(y_te.to_numpy(), p_te, thr)
    fig_threshold(y_tr.to_numpy(), oof, y_te.to_numpy(), p_te, thr)
    lc = fig_learning_curve(model, X_tr, y_tr)
    imp, coefs = fig_importance(model, X_te, y_te)
    mean_abs = fig_shap(model, X_tr, X_te)

    print("\nTop 10 — importância por permutação (queda de ROC-AUC no teste):")
    for _, r in imp.head(10).iterrows():
        print(f"  {r['feature']:<26} {r['mean']:+.4f} (+/-{r['std']:.4f})")

    print(f"\nRegressão Logística L1: {(coefs!=0).sum()} de {len(coefs)} coeficientes "
          f"não nulos ({(coefs==0).sum()} variáveis eliminadas)")
    print("\nMaiores coeficientes positivos (aumentam a chance de sobreviver):")
    for k, v in coefs.sort_values(ascending=False).head(6).items():
        print(f"  {k:<34} {v:+.4f}")
    print("Maiores coeficientes negativos (reduzem a chance de sobreviver):")
    for k, v in coefs.sort_values().head(6).items():
        print(f"  {k:<34} {v:+.4f}")

    if mean_abs is not None:
        print("\nTop 10 — |SHAP| médio:")
        for k, v in mean_abs.head(10).items():
            print(f"  {k:<34} {v:.4f}")

    ev.save_json(
        {
            "permutation_importance": imp.to_dict("records"),
            "coeficientes": coefs.to_dict(),
            "shap_mean_abs": (mean_abs.to_dict() if mean_abs is not None else None),
            "learning_curve": lc,
        },
        cfg.ARTIFACTS_DIR / "interpretation.json",
    )
    print("\nInterpretação salva em artifacts/interpretation.json")


if __name__ == "__main__":
    main()

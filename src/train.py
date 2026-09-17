"""
train.py — Treinamento, seleção de modelo e ajuste de hiperparâmetros.

Executar: python -m src.train

Protocolo anti-vazamento aplicado:
  1. Colunas pós-desfecho removidas ANTES de qualquer divisão (src/data.py).
  2. Divisão hold-out estratificada 80/20 feita ANTES de qualquer ajuste.
  3. Todo pré-processamento ocorre dentro do Pipeline -> refeito em cada fold.
  4. Seleção de modelo e busca de hiperparâmetros usam apenas o TREINO,
     via StratifiedKFold de 5 folds.
  5. O limiar de decisão é calibrado com cross_val_predict no TREINO.
  6. O conjunto de teste é tocado uma única vez, no fim, apenas para reportar.
"""

from __future__ import annotations

import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    cross_validate,
)

from . import config as cfg
from . import data as data_mod
from . import evaluation as ev
from .models import get_candidates

warnings.filterwarnings("ignore")

FRESH = False   # sobrescrito por --fresh na linha de comando

# Triagem: CV simples (barata). Ajuste e comparação final: CV repetida.
CV = StratifiedKFold(n_splits=cfg.N_SPLITS, shuffle=True, random_state=cfg.RANDOM_STATE)
RCV = RepeatedStratifiedKFold(n_splits=cfg.N_SPLITS, n_repeats=cfg.N_REPEATS,
                              random_state=cfg.RANDOM_STATE)
RCV_FINAL = RepeatedStratifiedKFold(n_splits=cfg.N_SPLITS, n_repeats=cfg.N_REPEATS_FINAL,
                                    random_state=cfg.RANDOM_STATE)


def _hr(title):
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)


# ---------------------------------------------------------------------------
def screen_candidates(X_tr, y_tr) -> pd.DataFrame:
    """Etapa 1 — triagem dos candidatos com hiperparâmetros padrão (5-fold CV)."""
    _hr("ETAPA 1 — TRIAGEM DE CANDIDATOS (5-fold CV no TREINO)")
    rows = []
    for name, spec in get_candidates().items():
        t0 = time.time()
        scores = cross_validate(
            spec["pipeline"], X_tr, y_tr, cv=CV,
            scoring=["roc_auc", "average_precision", "balanced_accuracy", "f1"],
            n_jobs=-1, return_train_score=True,
        )
        rows.append(
            {
                "modelo": name,
                "cv_roc_auc": scores["test_roc_auc"].mean(),
                "cv_roc_auc_std": scores["test_roc_auc"].std(),
                "train_roc_auc": scores["train_roc_auc"].mean(),
                "gap": scores["train_roc_auc"].mean() - scores["test_roc_auc"].mean(),
                "cv_pr_auc": scores["test_average_precision"].mean(),
                "cv_bal_acc": scores["test_balanced_accuracy"].mean(),
                "cv_f1": scores["test_f1"].mean(),
                "tempo_s": time.time() - t0,
                "nota": spec["note"],
            }
        )
        r = rows[-1]
        print(f"  {name:<30} ROC-AUC CV = {r['cv_roc_auc']:.4f} "
              f"(+/-{r['cv_roc_auc_std']:.4f})  treino = {r['train_roc_auc']:.4f}  "
              f"gap = {r['gap']:+.4f}   [{r['tempo_s']:.1f}s]")

    return pd.DataFrame(rows).sort_values("cv_roc_auc", ascending=False)


def tune(name: str, X_tr, y_tr):
    """Etapa 2 — busca de hiperparâmetros do candidato escolhido."""
    _hr(f"ETAPA 2 — AJUSTE DE HIPERPARÂMETROS: {name}")
    spec = get_candidates()[name]
    grid = spec["grid"]
    n_comb = int(np.prod([len(v) for v in grid.values()])) if grid else 1
    print(f"  Espaço de busca: {n_comb} combinações")

    if n_comb <= 30:
        search = GridSearchCV(
            spec["pipeline"], grid, cv=RCV, scoring=cfg.SCORING,
            n_jobs=-1, refit=True, return_train_score=True,
        )
        print(f"  Estratégia: GridSearchCV exaustiva, CV repetida "
              f"{cfg.N_SPLITS}x{cfg.N_REPEATS}")
    else:
        search = RandomizedSearchCV(
            spec["pipeline"], grid, n_iter=12, cv=RCV, scoring=cfg.SCORING,
            n_jobs=-1, refit=True, random_state=cfg.RANDOM_STATE,
            return_train_score=True,
        )
        print(f"  Estratégia: RandomizedSearchCV 12 amostras, CV repetida "
              f"{cfg.N_SPLITS}x{cfg.N_REPEATS}")

    t0 = time.time()
    search.fit(X_tr, y_tr)
    print(f"  Concluído em {time.time()-t0:.1f}s")
    print(f"  Melhor ROC-AUC CV : {search.best_score_:.4f}")
    print("  Melhores parâmetros:")
    for k, v in sorted(search.best_params_.items()):
        print(f"      {k} = {v}")
    return search


def calibrate_threshold(model, X_tr, y_tr) -> tuple[float, np.ndarray]:
    """
    Etapa 3 — limiar de decisão calibrado por validação cruzada no TREINO.

    Usar o teste aqui seria vazamento: o limiar é um hiperparâmetro.
    """
    _hr("ETAPA 3 — CALIBRAÇÃO DO LIMIAR DE DECISÃO (no TREINO, via CV)")
    oof = cross_val_predict(model, X_tr, y_tr, cv=CV, method="predict_proba", n_jobs=-1)[:, 1]
    thr_f1 = ev.best_threshold(y_tr, oof, "f1")
    thr_yd = ev.best_threshold(y_tr, oof, "youden")

    print(f"  Limiar padrão  0.500 -> F1 out-of-fold = {ev.compute_metrics(y_tr, oof, 0.5)['f1']:.4f}")
    print(f"  Limiar máx-F1  {thr_f1:.3f} -> F1 out-of-fold = {ev.compute_metrics(y_tr, oof, thr_f1)['f1']:.4f}")
    print(f"  Limiar Youden  {thr_yd:.3f} -> bal.acc out-of-fold = "
          f"{ev.compute_metrics(y_tr, oof, thr_yd)['balanced_accuracy']:.4f}")
    print(f"\n  Escolhido: {thr_f1:.3f} (máximo F1 out-of-fold)")
    return thr_f1, oof


# ---------------------------------------------------------------------------
def main():
    _hr("TREINAMENTO — PROGNÓSTICO DE SOBREVIDA EM CÂNCER DE PULMÃO")

    df = data_mod.load_raw(verbose=False)
    print("Harness de validação do dataset bruto: OK (9 grupos de regras)")

    X, y = data_mod.make_xy(df, include_leaky=False)
    print(f"\nMatriz de features: {X.shape[0]} x {X.shape[1]} colunas")
    print(f"Colunas removidas por vazamento/redundância ({len(cfg.DROP_ALWAYS)}): "
          f"{', '.join(cfg.DROP_ALWAYS)}")

    X_tr, X_te, y_tr, y_te = data_mod.split(X, y)
    print(f"\nDivisão hold-out estratificada 80/20:")
    print(f"  treino = {len(X_tr)} (positivos {y_tr.mean():.1%})")
    print(f"  teste  = {len(X_te)} (positivos {y_te.mean():.1%})")

    # ---- Etapa 1
    screening = screen_candidates(X_tr, y_tr)
    screening.to_csv(cfg.ARTIFACTS_DIR / "screening.csv", index=False)

    # ---- Etapa 2 — ajusta TODOS os candidatos com grade (a ordem da triagem
    #       usa hiperparâmetros padrão e pode inverter depois do tuning)
    #
    # Checkpoint: a busca é a etapa cara. O resultado é persistido para que a
    # análise posterior possa ser refeita sem repetir o ajuste.
    ckpt = cfg.ARTIFACTS_DIR / "searches.joblib"
    if ckpt.exists() and not FRESH:
        searches = joblib.load(ckpt)
        _hr("ETAPA 2 — AJUSTE DE HIPERPARÂMETROS (recuperado do checkpoint)")
        for n, s in searches.items():
            print(f"  {n:<30} melhor ROC-AUC CV = {s.best_score_:.4f}  {s.best_params_}")
        print("\n  Use --fresh para refazer a busca do zero.")
    else:
        searches = {}
        for name, spec in get_candidates().items():
            if not spec["grid"]:
                continue
            searches[name] = tune(name, X_tr, y_tr)
        joblib.dump(searches, ckpt)
        print(f"\nCheckpoint salvo em {ckpt.relative_to(cfg.ROOT)}")

    _hr(f"ETAPA 2b — COMPARAÇÃO FINAL COM ERRO PADRÃO "
        f"(CV repetida {cfg.N_SPLITS}x{cfg.N_REPEATS_FINAL} = "
        f"{cfg.N_SPLITS*cfg.N_REPEATS_FINAL} estimativas)")
    rows = []
    for n, s in searches.items():
        print(f"  avaliando {n}...", flush=True)
        scores = cross_val_score(
            clone(s.best_estimator_), X_tr, y_tr, cv=RCV_FINAL,
            scoring=cfg.SCORING, n_jobs=1,
        )
        rows.append(
            {
                "modelo": n,
                "roc_auc": float(scores.mean()),
                "erro_padrao": float(scores.std(ddof=1) / np.sqrt(len(scores))),
                "cv_triagem": float(
                    screening.loc[screening["modelo"] == n, "cv_roc_auc"].iloc[0]
                ),
            }
        )
    tuned = pd.DataFrame(rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)

    top = tuned.iloc[0]
    limite = top["roc_auc"] - top["erro_padrao"]
    tuned["dentro_de_1EP"] = tuned["roc_auc"] >= limite
    for _, r in tuned.iterrows():
        marca = "  (empate técnico)" if r["dentro_de_1EP"] and r["modelo"] != top["modelo"] else ""
        print(f"  {r['modelo']:<30} {r['roc_auc']:.4f} +/- {r['erro_padrao']:.4f}{marca}")

    print(f"\n  Regra de decisão: entre os modelos dentro de 1 erro padrão do melhor")
    print(f"  ({limite:.4f}), escolhe-se o de maior média. Diferenças menores que")
    print(f"  o erro padrão são ruído da estimativa, não evidência de superioridade.")
    tuned.to_csv(cfg.ARTIFACTS_DIR / "tuned_comparison.csv", index=False)

    best_name = top["modelo"]
    search = searches[best_name]
    best_model = search.best_estimator_
    print(f"\n  >> Modelo final: {best_name}")
    print(f"     ROC-AUC = {top['roc_auc']:.4f} +/- {top['erro_padrao']:.4f}")
    pd.DataFrame(search.cv_results_).to_csv(cfg.ARTIFACTS_DIR / "cv_results.csv", index=False)

    # ---- Etapa 3
    threshold, oof = calibrate_threshold(best_model, X_tr, y_tr)

    # ---- Etapa 4 — avaliação final
    _hr("ETAPA 4 — AVALIAÇÃO FINAL (o teste é usado UMA única vez)")
    p_tr = best_model.predict_proba(X_tr)[:, 1]
    p_te = best_model.predict_proba(X_te)[:, 1]

    m_train = ev.compute_metrics(y_tr, p_tr, threshold)
    m_oof = ev.compute_metrics(y_tr, oof, threshold)
    m_test = ev.compute_metrics(y_te, p_te, threshold)

    table = ev.metrics_table(
        {"Treino (reajuste)": m_train, "Validação cruzada (OOF)": m_oof, "Teste (hold-out)": m_test}
    )
    print("\n" + table.round(4).to_string())
    print(f"\n  Gap ROC-AUC treino - teste: {m_train['roc_auc'] - m_test['roc_auc']:+.4f}")
    print(f"  Gap ROC-AUC   OOF  - teste: {m_oof['roc_auc'] - m_test['roc_auc']:+.4f}")

    print("\n" + ev.text_report(y_te, p_te, threshold))

    # ---- Etapa 5 — robustez temporal
    _hr("ETAPA 5 — TESTE DE ROBUSTEZ: DIVISÃO TEMPORAL")
    Xt_tr, Xt_te, yt_tr, yt_te = data_mod.temporal_split(df)

    tmodel = clone(best_model).fit(Xt_tr, yt_tr)
    pt = tmodel.predict_proba(Xt_te)[:, 1]
    m_temp = ev.compute_metrics(yt_te, pt, threshold)
    print(f"  Treino: diagnósticos < {cfg.TEMPORAL_CUTOFF_YEAR} (n={len(Xt_tr)})")
    print(f"  Teste : diagnósticos >= {cfg.TEMPORAL_CUTOFF_YEAR} (n={len(Xt_te)})")
    print(f"  ROC-AUC = {m_temp['roc_auc']:.4f} | F1 = {m_temp['f1']:.4f} "
          f"| bal.acc = {m_temp['balanced_accuracy']:.4f}")

    # ---- Persistência
    joblib.dump(
        {
            "model": best_model,
            "threshold": threshold,
            "model_name": best_name,
            "best_params": search.best_params_,
            "features": list(X.columns),
            "sklearn_version": __import__("sklearn").__version__,
        },
        cfg.MODELS_DIR / "model.joblib",
    )
    ev.save_json(
        {
            "model_name": best_name,
            "best_params": search.best_params_,
            "threshold": threshold,
            "cv_roc_auc": float(top["roc_auc"]),
            "cv_erro_padrao": float(top["erro_padrao"]),
            "metrics": {
                "train": m_train, "cv_oof": m_oof,
                "test": m_test, "temporal_test": m_temp,
            },
            "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
            "n_features_raw": int(X.shape[1]),
            "dropped_columns": cfg.DROP_ALWAYS,
        },
        cfg.ARTIFACTS_DIR / "results.json",
    )
    np.save(cfg.ARTIFACTS_DIR / "proba_test.npy", p_te)
    np.save(cfg.ARTIFACTS_DIR / "y_test.npy", y_te.to_numpy())
    np.save(cfg.ARTIFACTS_DIR / "proba_oof.npy", oof)
    np.save(cfg.ARTIFACTS_DIR / "y_train.npy", y_tr.to_numpy())
    np.save(cfg.ARTIFACTS_DIR / "proba_train.npy", p_tr)

    print(f"\nModelo salvo em models/model.joblib")
    print(f"Resultados salvos em artifacts/results.json")
    return best_model


if __name__ == "__main__":
    import sys

    FRESH = "--fresh" in sys.argv
    main()

"""
predict.py — Demonstração de funcionamento do modelo treinado.

Executar:
    python -m src.predict                      # demonstração completa
    python -m src.predict --csv arquivo.csv    # prediz um lote novo

Mostra:
  1. Predições em pacientes reais do conjunto de teste, com acerto/erro.
  2. Predições em três casos clínicos sintéticos construídos à mão.
  3. Como o escore varia ao mudar uma única variável (análise what-if).
  4. Exportação das predições do teste para CSV.
"""

from __future__ import annotations

import sys
import warnings

import joblib
import numpy as np
import pandas as pd

from . import config as cfg
from . import data as data_mod

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)


def load_model():
    bundle = joblib.load(cfg.MODELS_DIR / "model.joblib")
    return bundle["model"], bundle["threshold"], bundle


def predict_frame(model, thr, X: pd.DataFrame) -> pd.DataFrame:
    p = model.predict_proba(X)[:, 1]
    return pd.DataFrame(
        {
            "prob_sobrevida": p.round(4),
            "predicao": np.where(p >= thr, "Sobrevive", "Não sobrevive"),
            "confianca": np.where(
                np.abs(p - thr) > 0.25, "alta",
                np.where(np.abs(p - thr) > 0.10, "média", "baixa"),
            ),
        },
        index=X.index,
    )


# ---------------------------------------------------------------------------
def demo_test_patients(model, thr, X_te, y_te, df, n=12):
    print("=" * 90)
    print("1. PREDICOES EM PACIENTES REAIS DO CONJUNTO DE TESTE")
    print("=" * 90)
    out = predict_frame(model, thr, X_te)
    out["real"] = np.where(y_te == 1, "Sobreviveu", "Não sobreviveu")
    out["acerto"] = np.where(
        (out["predicao"] == "Sobrevive") == (y_te == 1).to_numpy(), "OK", "ERRO"
    )
    ctx = df.loc[X_te.index, ["Patient_ID", "Age", "Gender", "Cancer_Stage",
                              "Metastasis", "Tumor_Size_cm"]]
    view = ctx.join(out)

    # amostra equilibrada: acertos e erros, alta e baixa confiança
    rng = np.random.default_rng(cfg.RANDOM_STATE)
    idx = list(view[view["acerto"] == "OK"].sample(n // 2, random_state=cfg.RANDOM_STATE).index)
    idx += list(view[view["acerto"] == "ERRO"].sample(n // 2, random_state=cfg.RANDOM_STATE).index)
    print(view.loc[idx].to_string(index=False))

    acc = (out["acerto"] == "OK").mean()
    print(f"\nAcurácia no conjunto de teste completo ({len(X_te)} pacientes): {acc:.1%}")
    print("Distribuição da confiança das predições:")
    print(out["confianca"].value_counts().to_string())
    return view


def demo_synthetic_cases(model, thr, df):
    print("\n" + "=" * 90)
    print("2. CASOS CLINICOS SINTETICOS")
    print("=" * 90)
    base = df.iloc[0].copy()

    casos = {
        "Caso A — rastreio precoce, não fumante": {
            "Age": 55, "Gender": "Female", "Smoking_Status": "Never Smoked",
            "Cigarettes_Per_Day": 0, "Years_Smoking": 0, "Pack_Years": 0.0,
            "Cancer_Stage": "Stage I", "Metastasis": "No", "Tumor_Size_cm": 1.8,
            "Diagnosis_Method": "LDCT Screening", "Cancer_Type": "NSCLC",
            "NSCLC_Subtype": "Adenocarcinoma", "Symptom_Count": 1,
            "Coughing_Blood": "No", "Chest_Pain": "No", "Fatigue": "No",
            "Air_Pollution_Exposure": "Low", "Alcohol_Use": "No Alcohol",
        },
        "Caso B — doença avançada, tabagista pesado": {
            "Age": 68, "Gender": "Male", "Smoking_Status": "Current Smoker",
            "Cigarettes_Per_Day": 30, "Years_Smoking": 40, "Pack_Years": 60.0,
            "Cancer_Stage": "Stage IV", "Metastasis": "Yes", "Tumor_Size_cm": 8.5,
            "Diagnosis_Method": "PET Scan", "Cancer_Type": "SCLC",
            "NSCLC_Subtype": "Not Applicable", "Symptom_Count": 8,
            "Coughing_Blood": "Yes", "Chest_Pain": "Yes", "Fatigue": "Yes",
            "Air_Pollution_Exposure": "High", "Alcohol_Use": "Heavy",
        },
        "Caso C — intermediário, quadro ambíguo": {
            "Age": 61, "Gender": "Male", "Smoking_Status": "Former Smoker",
            "Cigarettes_Per_Day": 15, "Years_Smoking": 20, "Pack_Years": 15.0,
            "Cancer_Stage": "Stage II", "Metastasis": "No", "Tumor_Size_cm": 4.2,
            "Diagnosis_Method": "CT Scan", "Cancer_Type": "NSCLC",
            "NSCLC_Subtype": "Squamous Cell", "Symptom_Count": 4,
            "Coughing_Blood": "No", "Chest_Pain": "Yes", "Fatigue": "Yes",
            "Air_Pollution_Exposure": "Moderate", "Alcohol_Use": "Moderate",
        },
    }

    rows = []
    for nome, campos in casos.items():
        r = base.copy()
        for k, v in campos.items():
            r[k] = v
        rows.append(r)
    X_new = pd.DataFrame(rows, index=list(casos))
    X_new, _ = data_mod.make_xy(
        X_new.assign(**{cfg.TARGET: "No"}), verbose=False
    )

    out = predict_frame(model, thr, X_new)
    for nome in casos:
        p = out.loc[nome, "prob_sobrevida"]
        barra = "#" * int(p * 40)
        print(f"\n  {nome}")
        print(f"    estádio {casos[nome]['Cancer_Stage']:<10} "
              f"tumor {casos[nome]['Tumor_Size_cm']} cm   "
              f"sintomas {casos[nome]['Symptom_Count']}")
        print(f"    probabilidade de sobrevida: {p:.3f}  |{barra:<40}|")
        print(f"    predição: {out.loc[nome,'predicao']}  "
              f"(confiança {out.loc[nome,'confianca']})")
    return X_new, out


def demo_whatif(model, thr, X_new):
    print("\n" + "=" * 90)
    print("3. ANALISE WHAT-IF — VARIANDO UMA UNICA VARIAVEL")
    print("=" * 90)
    base = X_new.loc[["Caso C — intermediário, quadro ambíguo"]]

    print("\n  Efeito do estadiamento (demais variáveis fixas no Caso C):")
    for stage in cfg.STAGE_ORDER:
        x = base.copy()
        x["Cancer_Stage"] = stage
        x["Metastasis"] = "Yes" if stage == "Stage IV" else "No"
        p = model.predict_proba(x)[0, 1]
        barra = "#" * int(p * 40)
        print(f"    {stage:<10} prob = {p:.3f}  |{barra:<40}|  "
              f"-> {'Sobrevive' if p >= thr else 'Não sobrevive'}")

    print("\n  Efeito da idade (Caso C, estádio II):")
    for age in [40, 50, 60, 70, 80]:
        x = base.copy()
        x["Age"] = age
        p = model.predict_proba(x)[0, 1]
        print(f"    {age} anos    prob = {p:.3f}")

    print("\n  Efeito da hemoptise (Coughing_Blood), Caso C:")
    for v in ["No", "Yes"]:
        x = base.copy()
        x["Coughing_Blood"] = v
        p = model.predict_proba(x)[0, 1]
        print(f"    Coughing_Blood = {v:<4} prob = {p:.3f}")


def export_predictions(model, thr, X_te, y_te, df):
    out = predict_frame(model, thr, X_te)
    out.insert(0, "Patient_ID", df.loc[X_te.index, "Patient_ID"])
    out["real"] = np.where(y_te == 1, "Sobreviveu", "Não sobreviveu")
    path = cfg.ARTIFACTS_DIR / "predicoes_teste.csv"
    out.to_csv(path, index=False)
    print(f"\n  Predições do teste exportadas -> {path.relative_to(cfg.ROOT)}")


# ---------------------------------------------------------------------------
def main():
    if "--csv" in sys.argv:
        path = sys.argv[sys.argv.index("--csv") + 1]
        model, thr, _ = load_model()
        df_new = pd.read_csv(path)
        X_new, _ = data_mod.make_xy(
            df_new.assign(**{cfg.TARGET: df_new.get(cfg.TARGET, "No")}), verbose=False
        )
        out = predict_frame(model, thr, X_new)
        print(out.to_string())
        return

    model, thr, bundle = load_model()
    print(f"Modelo carregado: {bundle['model_name']}")
    print(f"Hiperparâmetros : {bundle['best_params']}")
    print(f"Limiar          : {thr:.3f}")
    print(f"scikit-learn    : {bundle['sklearn_version']}\n")

    df = data_mod.load_raw(verbose=False)
    X, y = data_mod.make_xy(df, verbose=False)
    _, X_te, _, y_te = data_mod.split(X, y, verbose=False)

    demo_test_patients(model, thr, X_te, y_te, df)
    X_new, _ = demo_synthetic_cases(model, thr, df)
    demo_whatif(model, thr, X_new)
    export_predictions(model, thr, X_te, y_te, df)


if __name__ == "__main__":
    main()

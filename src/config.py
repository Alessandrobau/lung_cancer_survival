"""
config.py — Configuração central do projeto.

Contém, em um único lugar, o contrato de dados (schema esperado), a definição do
alvo, a política de exclusão de variáveis (vazamento / redundância) e os
parâmetros da estratégia experimental. Nenhum outro módulo define esses valores.
"""

from pathlib import Path

# ----------------------------------------------------------------------------
# Caminhos
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw" / "lung_cancer_dataset.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"
ARTIFACTS_DIR = ROOT / "artifacts"

for _d in (MODELS_DIR, REPORTS_DIR, FIG_DIR, ARTIFACTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# Reprodutibilidade
# ----------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20
N_SPLITS = 5          # StratifiedKFold para triagem
N_REPEATS = 3         # repetições no ajuste de hiperparâmetros
N_REPEATS_FINAL = 5   # repetições na comparação final entre famílias
SCORING = "roc_auc"   # métrica de seleção (ver justificativa no relatório)

# A triagem usa 5 folds simples (barata). O ajuste e a comparação final usam
# RepeatedStratifiedKFold: com 5 folds, o desvio entre folds (~0,034) é maior
# que a diferença entre hiperparâmetros (~0,003), e a escolha vira sorteio.
# Repetir a CV reduz o erro padrão da estimativa para ~0,005 e torna a
# comparação informativa.

# ----------------------------------------------------------------------------
# Alvo
# ----------------------------------------------------------------------------
TARGET = "Survived"
POSITIVE_LABEL = "Yes"   # classe positiva = paciente sobreviveu
TARGET_MAP = {"No": 0, "Yes": 1}

# ----------------------------------------------------------------------------
# POLÍTICA DE EXCLUSÃO DE VARIÁVEIS
# ----------------------------------------------------------------------------
# O problema é definido como PROGNÓSTICO NO MOMENTO DO DIAGNÓSTICO. Toda
# variável que só passa a existir DEPOIS desse instante é vazamento temporal e
# precisa sair antes de qualquer divisão de dados.

# (1) Vazamento de alvo — medido após o desfecho, não observável no diagnóstico.
LEAKY_OUTCOME = [
    "Survival_Months",   # tempo de sobrevida: é o próprio desfecho, em outra escala
]

# (2) Vazamento temporal — decisão clínica posterior ao instante de predição.
LEAKY_POSTERIOR = [
    "Treatment",         # tratamento só é definido após o diagnóstico; 'Palliative
                         # Care' codifica o prognóstico dado pelo próprio médico
]

# (3) Identificadores — sem poder preditivo, risco de memorização.
IDENTIFIERS = [
    "Patient_ID",
    "Diagnosis_Date",    # usada apenas para o teste de robustez temporal
]

# (4) Redundâncias determinísticas (verificadas em src/validate.py):
#     Age_Group      = discretização exata de Age
#     BMI_Category   = discretização exata de BMI
#     Country        = 60 níveis, 5 a 74 obs cada -> alta cardinalidade;
#                      WHO_Region preserva o sinal geográfico com 6 níveis
REDUNDANT = [
    "Age_Group",
    "BMI_Category",
    "Country",
]

DROP_ALWAYS = LEAKY_OUTCOME + LEAKY_POSTERIOR + IDENTIFIERS + REDUNDANT

# Variáveis derivadas exatas que SÃO mantidas (agregados clínicos úteis a
# modelos de árvore, que não somam colunas sozinhos). Documentadas por causa da
# colinearidade que introduzem em modelos lineares:
#     Pack_Years        = Cigarettes_Per_Day / 20 * Years_Smoking
#     Symptom_Count     = soma dos 10 sintomas binários
#     Risk_Factor_Count = soma dos 7 fatores de risco + (Air_Pollution == 'High')
DERIVED_KEPT = ["Pack_Years", "Symptom_Count", "Risk_Factor_Count"]

# ----------------------------------------------------------------------------
# Schema esperado (contrato de dados usado pelas validações)
# ----------------------------------------------------------------------------
NUMERIC_FEATURES = [
    "Diagnosis_Year",
    "Age",
    "Cigarettes_Per_Day",
    "Years_Smoking",
    "Pack_Years",
    "BMI",
    "Risk_Factor_Count",
    "Symptom_Count",
    "Tumor_Size_cm",
]

BINARY_FEATURES = [
    "Secondhand_Smoke", "Family_History", "Occupational_Hazard",
    "Chronic_Lung_Disease", "Asbestos_Exposure", "Radon_Exposure",
    "Previous_Cancer_History", "Coughing", "Shortness_of_Breath",
    "Chest_Pain", "Coughing_Blood", "Fatigue", "Weight_Loss", "Wheezing",
    "Recurrent_Infections", "Swallowing_Difficulty", "Finger_Clubbing",
    "Metastasis",
]

CATEGORICAL_FEATURES = [
    "WHO_Region", "Gender", "Smoking_Status", "Air_Pollution_Exposure",
    "Alcohol_Use", "Exercise_Frequency", "Genetic_Mutation", "Cancer_Type",
    "NSCLC_Subtype", "Cancer_Stage", "Diagnosis_Method",
]

# Ordem clínica do estadiamento — usada no encoding ordinal
STAGE_ORDER = ["Stage I", "Stage II", "Stage III", "Stage IV"]

# Faixas plausíveis para validação de sanidade dos dados
VALID_RANGES = {
    "Age": (18, 110),
    "BMI": (10.0, 70.0),
    "Tumor_Size_cm": (0.1, 30.0),
    "Cigarettes_Per_Day": (0, 100),
    "Years_Smoking": (0, 80),
    "Pack_Years": (0.0, 300.0),
    "Risk_Factor_Count": (0, 8),
    "Symptom_Count": (0, 10),
    "Diagnosis_Year": (1990, 2030),
}

# Corte temporal para o teste de robustez (split por ano de diagnóstico)
TEMPORAL_CUTOFF_YEAR = 2024   # treino: <= 2023 | teste: >= 2024

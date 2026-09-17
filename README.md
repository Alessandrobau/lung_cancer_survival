# Prognóstico de sobrevida em câncer de pulmão

Trabalho 1 — Inteligência Artificial II (2026/02) — Faculdade Antonio Meneghetti

Classificação binária supervisionada: estimar, **no momento do diagnóstico**, se um
paciente com câncer de pulmão sobreviverá. O foco metodológico do projeto é o
tratamento de vazamento de dados e a honestidade da estimativa de desempenho.

---

## Resultado em uma linha

Regressão Logística com regularização L1 (C = 0,1), **ROC-AUC de 0,778 no teste
hold-out** e 0,809 em divisão temporal. A análise de ablação mostra que o
estadiamento sozinho já entrega ROC-AUC 0,813 e que as outras 37 variáveis
agregam +0,0014 — menos de um terço do erro padrão da estimativa.

---

## Como reproduzir

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
bash run_all.sh
```

Ou etapa por etapa:

```bash
python -m src.eda                  # EDA + figuras 01-06
python -m src.leakage_experiment   # diagnóstico de vazamento + figura 07
python -m src.train --fresh        # treino, seleção de modelo, figuras de métrica
python -m src.interpret            # avaliação visual + SHAP, figuras 08-13
python -m src.ablation             # valor incremental por bloco, figura 14
python -m src.predict              # demonstração de funcionamento
```

Tudo é determinístico: `random_state = 42` em toda divisão, reamostragem e
estimador. Rodar duas vezes produz exatamente os mesmos números.

Tempo total em uma máquina de 1 núcleo: aproximadamente 8 minutos.
A busca de hiperparâmetros grava um checkpoint em `artifacts/searches.joblib`;
execuções seguintes de `python -m src.train` (sem `--fresh`) o reaproveitam.

---

## Estrutura

```
.
├── data/raw/lung_cancer_dataset.csv   dataset original, sem modificações
├── src/
│   ├── config.py             schema, política de exclusão de colunas, hiperparâmetros
│   ├── validate.py           harness: 9 grupos de regras que falham alto
│   ├── data.py               carga, remoção de vazamento, divisões
│   ├── features.py           ColumnTransformer (todo pré-processamento vive aqui)
│   ├── models.py             5 candidatos + grades de busca
│   ├── evaluation.py         métricas, limiar, relatórios
│   ├── train.py              triagem, ajuste, calibração de limiar, avaliação final
│   ├── interpret.py          curvas, importâncias, SHAP
│   ├── leakage_experiment.py ablação de vazamento + demonstração controlada
│   ├── ablation.py           valor incremental por bloco de variáveis
│   └── predict.py            demonstração de funcionamento e what-if
├── models/model.joblib       pipeline treinado + limiar + metadados
├── artifacts/                métricas em JSON/CSV, predições, checkpoint
└── reports/
    ├── relatorio.pdf         relatório técnico (entregável obrigatório)
    └── figures/              14 figuras geradas pelo código
```

---

## Decisões que definem o projeto

### 1. O problema é prognóstico *no diagnóstico*

Essa definição determina o que pode entrar no modelo. Três colunas foram
removidas antes de qualquer divisão de dados:

| Coluna | Motivo |
|---|---|
| `Survival_Months` | é o próprio desfecho em outra escala; 437 pacientes (21,9%) têm sobrevida maior que o seguimento fisicamente possível a partir do ano de diagnóstico |
| `Treatment` | decisão clínica posterior ao instante da predição; `Palliative Care` codifica o prognóstico do próprio médico |
| `Patient_ID`, `Diagnosis_Date` | identificadores |

Também saíram `Age_Group`, `BMI_Category` (discretizações exatas de colunas já
presentes) e `Country` (60 níveis, alguns com 5 observações; `WHO_Region`
preserva o sinal geográfico).

### 2. Todo pré-processamento vive dentro do `Pipeline`

Médias, desvios, medianas e vocabulários de categorias são aprendidos apenas nos
folds de treino. `src/leakage_experiment.py` mede o custo de errar isso: com 3000
atributos de ruído puro e rótulos aleatórios (AUC verdadeiro = 0,500), ajustar a
seleção de atributos no dataset inteiro produz AUC aparente de **0,676**; a mesma
seleção dentro do `Pipeline` devolve 0,490.

### 3. A seleção de modelo usa validação cruzada repetida

Com 5 folds simples, o desvio entre folds (±0,034) é uma ordem de grandeza maior
que a diferença entre hiperparâmetros (~0,003) — a escolha vira sorteio. A
primeira versão deste projeto caiu nessa armadilha: escolheu C = 0,03, que zera
61 de 63 coeficientes e reduz o modelo a uma única variável. Com
`RepeatedStratifiedKFold` (5×5), o erro padrão cai para ~0,005 e a escolha passa
a ser informativa.

### 4. O limiar de decisão é calibrado no treino

O limiar (0,464) é um hiperparâmetro e foi escolhido por `cross_val_predict` no
conjunto de treino. Ajustá-lo olhando o teste seria vazamento.

### 5. O conjunto de teste é tocado uma única vez

Apenas ao final, para reportar. Nenhuma decisão do projeto foi tomada olhando
para ele.

---

## Validações automáticas (harness)

`src/validate.py` roda antes de todo treinamento e interrompe a execução se
alguma premissa falhar:

| Regra | Verifica |
|---|---|
| R1 | schema: colunas esperadas presentes, tipos corretos |
| R2 | alvo binário, sem ausentes, classe minoritária > 5% |
| R3 | zero valores nulos |
| R4 | `Patient_ID` único, zero linhas duplicadas |
| R5 | numéricas dentro de faixas clinicamente plausíveis |
| R6 | categorias dentro do vocabulário esperado |
| R7 | coerência clínica (não fumante ⇒ carga tabágica zero; SCLC ⇒ subtipo N/A; Stage IV ⇒ metástase) |
| R8 | nenhuma coluna proibida sobrevive ao pré-processamento |
| R9 | agregados continuam sendo função exata das origens |

---

## Limitação principal

O dataset é sintético e tem pouco sinal fora do estadiamento. Dois coeficientes
do modelo final têm sinal clinicamente implausível: maior exposição à poluição e
maior idade **aumentam** a probabilidade prevista de sobrevida. São artefatos da
geração dos dados, não achados. O relatório em `reports/relatorio.pdf` discute
isso em detalhe.

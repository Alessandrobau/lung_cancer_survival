# Prognóstico de Sobrevida em Câncer de Pulmão

Análise de aprendizado de máquina para estimar, no momento do diagnóstico, se um paciente com câncer de pulmão sobreviverá. O projeto utiliza uma **Árvore de Decisão** por sua interpretabilidade: suas regras podem ser lidas diretamente na árvore treinada.

## Objetivos

- Explorar a relação entre características clínicas e a sobrevida.
- Evitar vazamento de dados na preparação e avaliação do modelo.
- Comparar o desempenho da árvore com um baseline.
- Ajustar a complexidade da árvore por validação cruzada.
- Avaliar o modelo em um conjunto de teste reservado.
- Investigar se `Cancer_Stage` concentra a maior parte do sinal preditivo.

## Dataset

O notebook baixa o dataset do Kaggle:

`laxmikantaroy/lung-cancer-risk-factors-and-survival-dataset`

O arquivo esperado é `lung_cancer_dataset.csv`. A variável-alvo é `Survived`, com as classes:

- `No`: não sobreviveu
- `Yes`: sobreviveu

As colunas `Survival_Months` e `Treatment` são removidas por representarem informações posteriores ao diagnóstico ou o próprio desfecho. Também são excluídos identificadores e variáveis redundantes antes da divisão entre treino e teste.

## Metodologia

1. Carregamento e validação do schema, valores ausentes, duplicatas e faixas plausíveis.
2. Análise exploratória da distribuição do alvo e da sobrevida por estágio.
3. Divisão estratificada em treino e teste, com 20% reservado para avaliação final.
4. Pré-processamento dentro de um `Pipeline`:
   - variáveis numéricas sem escala;
   - variáveis binárias e ordinais com `OrdinalEncoder`;
   - variáveis nominais com `OneHotEncoder`.
5. Validação cruzada estratificada com 5 folds e `ROC-AUC` como métrica principal.
6. Ajuste de `max_depth`, `min_samples_leaf` e `criterion` com `GridSearchCV`.
7. Avaliação final usando ROC-AUC, acurácia, acurácia balanceada, precisão, revocação, F1 e matriz de confusão.
8. Comparação exploratória com Random Forest, Gradient Boosting e Regressão Logística.

## Como executar

### Requisitos

- Python 3.10 ou superior
- Jupyter Notebook ou Visual Studio Code com a extensão Jupyter
- Acesso à internet para o download via KaggleHub

Instale as dependências:

```bash
pip install kagglehub numpy pandas matplotlib scipy scikit-learn jupyter
```

Execute todas as células de `notebook_final.ipynb` na ordem apresentada. O notebook fará o download do dataset automaticamente e exibirá as tabelas, gráficos, métricas e a árvore final.

Caso o CSV já esteja disponível localmente, altere `CSV_PATH` na primeira célula para apontar para o arquivo.

## Estrutura

```text
.
├── notebook_final.ipynb
└── README.md
```

## Observações e limitações

Este projeto tem finalidade acadêmica e exploratória. Os resultados não devem ser usados para decisões clínicas. A qualidade das conclusões depende da representatividade, da qualidade e da origem do dataset; validação externa e dados clínicos reais seriam necessários antes de qualquer uso prático.

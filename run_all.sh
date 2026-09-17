#!/usr/bin/env bash
# Reproduz o projeto inteiro do zero, na ordem correta.
set -e
echo ">>> 1/5  Análise exploratória"
python -m src.eda
echo ">>> 2/5  Diagnóstico de vazamento de dados"
python -m src.leakage_experiment
echo ">>> 3/5  Treinamento e seleção de modelo"
python -m src.train --fresh
echo ">>> 4/5  Avaliação visual e interpretação (SHAP)"
python -m src.interpret
echo ">>> 5/5  Ablação por blocos e demonstração"
python -m src.ablation
python -m src.predict
echo ">>> Concluído. Figuras em reports/figures/, métricas em artifacts/."

echo ">>> Extra: regenerar o PDF do relatório (requer wkhtmltopdf)"
if command -v wkhtmltopdf >/dev/null 2>&1; then
  wkhtmltopdf --enable-local-file-access --page-size A4 \
    --margin-top 16mm --margin-bottom 14mm --margin-left 14mm --margin-right 14mm \
    --encoding utf-8 --quiet reports/relatorio.html reports/relatorio.pdf
  echo "    reports/relatorio.pdf atualizado"
else
  echo "    wkhtmltopdf não encontrado; PDF não regenerado"
fi

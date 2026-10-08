# Política de concessão de crédito

Proposta de regra para decidir **quem recebe o empréstimo e qual o valor máximo**, a partir das
bases concessao e inadimplência do case.

**A regra:** O score diz quem é o cliente, a renda diz quanto cabe no orçamento.

| Indicador (teste fora do tempo, jan–jul/2021) | Política atual | Proposta |
|---|---|---|
| Contratos aprovados integralmente | 100% | 71,3% |
| Inadimplência em 12 meses | 12,4% | 3,5% |
| Resultado econômico simplificado (LGD 60%) | R$ 2,26 mi | R$ 5,35 mi |
| Com contraproposta de valor menor (hipótese) | – | R$ 6,65 mi |

## Estrutura

```
data/                 concessao.csv e inadimplencia.csv
notebooks/
  relatorio.ipynb     relatório: narrativa, código comentado, gráficos e tabelas
src/
  dados.py            carga, limpeza e construção do alvo
  features.py         variáveis derivadas e separação treino x teste
  scorecard.py        WoE, IV, regressão logística, pontos, faixas e métricas
  politica.py         matriz de decisão, valor máximo e simulação de impacto
  graficos.py         gráficos do relatório
figuras/              gráficos exportados para os slides
```

## Como reproduzir

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace notebooks/relatorio.ipynb
jupyter nbconvert --to html notebooks/relatorio.ipynb
```

O relatório roda do início ao fim em menos de um minuto. Semente fixa em 42.

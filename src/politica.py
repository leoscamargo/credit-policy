# Política de concessão: resultado por contrato, matriz de decisão, valor máximo e impacto.
import numpy as np

import pandas as pd

# Premissas financeiras (o case não informa)
LGD = 0.60             # perda dado o default: % do valor emprestado que o banco perde
CUSTO_CAPTACAO = 0.01  # custo do dinheiro para o banco, ao mês

LIMITES_COMPROMETIMENTO = [0, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, np.inf]
ROTULOS_COMPROMETIMENTO = ["≤20%", "20-25%", "25-30%", "30-35%", "35-40%", "40-50%", ">50%"]
DEGRAU = 0.30          # ponto em que a inadimplência salta (achado da análise exploratória)
ORDEM_FAIXAS = ["A", "B", "C", "D", "E"]

def faixa_comprometimento(comprometimento: pd.Series) -> pd.Series:
    return pd.cut(
        comprometimento, 
        bins=LIMITES_COMPROMETIMENTO, 
        labels=ROTULOS_COMPROMETIMENTO
    )

# Resultado econômico simplificado de um contrato, em R$.
# Bom: recebe todas as parcelas e devolve o valor captado com o custo do dinheiro.
# Mau: perde LGD x valor emprestado (premissa conservadora: nenhuma parcela paga).
def resultado_contrato(valor, parcela, prazo, mau, lgd: float = LGD,
    custo: float = CUSTO_CAPTACAO):
    ganho_se_bom = parcela * prazo - valor * (1 + custo) ** prazo
    perda_se_mau = -lgd * valor
    return np.where(np.asarray(mau) == 1, 
        perda_se_mau, ganho_se_bom
    )

# Cruza faixa de risco x comprometimento. medida: 'margem', 'maus' ou 'contratos'.
def matriz(df: pd.DataFrame, medida: str = "margem", lgd: float = LGD) -> pd.DataFrame:
    d = df.assign(
        faixa_comp=faixa_comprometimento(df["comprometimento"]),
        resultado=resultado_contrato(df["valor_emprestado"], 
                                    df["valor_parcela"],
                                    df["valor_prazo"], 
                                    df["mau"], lgd)
        )
    g = d.groupby(["faixa", "faixa_comp"], observed=False)
    if medida == "margem":
        tabela = g["resultado"].sum() / g["valor_emprestado"].sum()
    elif medida == "maus":
        tabela = g["mau"].mean()
    else:
        tabela = g["mau"].size()
    return tabela.unstack("faixa_comp").reindex(ORDEM_FAIXAS)

# Teto de comprometimento por faixa, aprendido no treino.
# Se os contratos da faixa com parcela acima do degrau (30% da renda) não entregam a margem
# mínima, a faixa ganha teto no degrau. Caso contrário, não há teto observado nos dados.

def derivar_tetos(treino: pd.DataFrame, margem_minima: float = 0.03, lgd: float = LGD) -> dict:
    acima = treino[treino["comprometimento"] > DEGRAU]
    res = resultado_contrato(acima["valor_emprestado"], 
                            acima["valor_parcela"],
                            acima["valor_prazo"], 
                            acima["mau"], lgd)
    margem = (pd.Series(res, index=acima.index).groupby(acima["faixa"], 
                            observed=False).sum() / acima.groupby("faixa", observed=False)["valor_emprestado"].sum())
    return {f: (DEGRAU if margem[f] < margem_minima else None) for f in ORDEM_FAIXAS}

# Maior empréstimo cuja parcela (tabela Price) cabe em teto x renda.
def valor_maximo(renda, taxa, prazo, teto):
    parcela_max = renda * teto
    return parcela_max * (1 - (1 + taxa) ** (-prazo)) / taxa

# Decisão por contrato: aprova o valor pedido ou faz contraproposta de valor menor.
def aplicar_politica(df: pd.DataFrame, tetos: dict) -> pd.DataFrame:
    d = df.copy()
    d["teto"] = d["faixa"].astype(str).map(tetos).astype(float)  # None vira NaN = sem teto
    dentro = d["teto"].isna() | (d["comprometimento"] <= d["teto"])
    d["decisao"] = np.where(dentro, "aprova", "contraproposta")
    d["valor_aprovado"] = np.where(
        dentro, d["valor_emprestado"],
        valor_maximo(d["valor_renda"], d["valor_taxa"], d["valor_prazo"], d["teto"]))
    d["parcela_aprovada"] = np.where(dentro, d["valor_parcela"], d["valor_renda"] * d["teto"])
    return d

# Taxa de maus (treino) de cada faixa entre contratos com parcela dentro do teto."""
def risco_dentro_do_teto(treino: pd.DataFrame, tetos: dict) -> dict:
    risco = {}
    for faixa, teto in tetos.items():
        d = treino[treino["faixa"] == faixa]
        if teto is not None:
            d = d[d["comprometimento"] <= teto]
        risco[faixa] = float(d["mau"].mean())
    return risco

# Compara a política atual (aprovou todos) com a proposta, no período de teste.
# 'medido': só contratos aprovados integralmente, com o desfecho que realmente ocorreu.
# 'com contraproposta': soma o resultado ESPERADO dos contratos reduzidos, supondo que o
# cliente aceita o valor menor e passa a ter o risco da faixa dentro do teto (hipótese).
def simular_impacto(teste: pd.DataFrame, treino: pd.DataFrame, tetos: dict,
                lgd: float = LGD, custo: float = CUSTO_CAPTACAO) -> dict:
    d = aplicar_politica(teste, tetos)
    res_real = resultado_contrato(d["valor_emprestado"], d["valor_parcela"], d["valor_prazo"],
                            d["mau"], lgd, custo)
    integral = d["decisao"] == "aprova"
    red = d[~integral]
    p_mau = red["faixa"].astype(str).map(risco_dentro_do_teto(treino, tetos)).astype(float)
    ganho = red["parcela_aprovada"] * red["valor_prazo"] \
        - red["valor_aprovado"] * (1 + custo) ** red["valor_prazo"]
    res_esperado = (1 - p_mau) * ganho + p_mau * (-lgd * red["valor_aprovado"])
    return {
        "contratos": len(d),
        "valor_pedido": float(d["valor_emprestado"].sum()),
        "atual_maus": float(d["mau"].mean()),
        "atual_resultado": float(res_real.sum()),
        "aprovados_integral": float(integral.mean()),
        "proposta_maus": float(d.loc[integral, "mau"].mean()),
        "proposta_resultado_medido": float(res_real[integral].sum()),
        "contratos_reduzidos": int((~integral).sum()),
        "valor_reduzido_medio": float((red["valor_aprovado"] / red["valor_emprestado"]).mean())
        if len(red) else np.nan,
        "proposta_resultado_com_contraproposta": float(res_real[integral].sum()
                                                    + res_esperado.sum()),
    }
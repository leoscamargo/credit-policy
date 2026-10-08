# Variáveis derivadas e separação treino x teste fora do tempo.
import pandas as pd

# Variáveis que descrevem o CLIENTE: entram no score de perfil
VARS_PERFIL = ["idade", "tempo_conta", "valor_restritivos", "restr_renda",
                "cat_escolaridade", "flag_cliente_ativo"]
# Variáveis que descrevem o EMPRÉSTIMO: ficam fora do score e entram na capacidade de pagamento
VARS_EMPRESTIMO = ["comprometimento", "emp_renda", "valor_emprestado", "valor_prazo"]
# Avaliadas na análise exploratória, mas fora da decisão (ver relatório)
VARS_DESCARTADAS = ["valor_renda", "valor_taxa", "cod_agencia"]

CATEGORICAS = ["cat_escolaridade", "flag_cliente_ativo"]
FIM_TREINO = "2020-12-31"


def criar_variaveis(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["idade"] = (df["data_concessao"] - df["data_nascimento"]).dt.days / 365.25
    df["tempo_conta"] = (df["data_concessao"] - df["data_abertura_conta"]).dt.days / 30.44
    df["comprometimento"] = df["valor_parcela"] / df["valor_renda"]
    df["restr_renda"] = df["valor_restritivos"] / df["valor_renda"]
    df["emp_renda"] = df["valor_emprestado"] / df["valor_renda"]
    return df


# Treino = passado; teste = período seguinte (fora do tempo), como no uso real."""
def separar_treino_teste(base: pd.DataFrame, fim_treino: str = FIM_TREINO):
    treino = base[base["data_concessao"] <= fim_treino].copy()
    teste = base[base["data_concessao"] > fim_treino].copy()
    return treino, teste

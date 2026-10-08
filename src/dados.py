# Carga, limpeza e construção da variável alvo (bom x mau pagador).
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATAS_CONCESSAO = ["data_concessao", "data_nascimento", "data_abertura_conta"]
MOB_ALVO = 12  # mau pagador = inadimplente 12 meses após a concessão (dica do case)

# Lê o CSV em UTF-8; se os acentos falharem, tenta latin-1."""
def _ler_csv(caminho: Path, **kwargs) -> pd.DataFrame:
    try:
        return pd.read_csv(caminho, encoding="utf-8", **kwargs)
    except UnicodeDecodeError:
        return pd.read_csv(caminho, encoding="latin-1", **kwargs)


# "Fundamental Incompleto" aparece duas vezes (uma com espaço no fim), nulo vira categoria
def carregar_concessao(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    conc = _ler_csv(data_dir / "concessao.csv", parse_dates=DATAS_CONCESSAO)
    conc["cat_escolaridade"] = conc["cat_escolaridade"].str.strip().fillna("Nulo")
    assert conc["id_cliente"].is_unique, "esperado 1 contrato por cliente"
    return conc


def carregar_inadimplencia(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    return _ler_csv(data_dir / "inadimplencia.csv", parse_dates=["data_referencia"])


# MOB (month on book) = meses entre a concessão e a data de referência.
def adicionar_mob(inad: pd.DataFrame, conc: pd.DataFrame) -> pd.DataFrame:
    inad = inad.merge(conc[["id_cliente", "data_concessao"]], on="id_cliente", how="left")
    inad["mob"] = ((inad["data_referencia"].dt.year - inad["data_concessao"].dt.year) * 12
                + (inad["data_referencia"].dt.month - inad["data_concessao"].dt.month))
    return inad


# Mantém só safras maduras e marca mau = inadimplente no MOB alvo.
# Safra madura: concedida até (data de extração - mob_alvo meses).
# Sem registro no MOB alvo = quitou antes = bom pagador (mau = 0).
def construir_base_modelagem(conc: pd.DataFrame, inad: pd.DataFrame,
                            mob_alvo: int = MOB_ALVO) -> pd.DataFrame:
    inad = adicionar_mob(inad, conc)
    data_extracao = inad["data_referencia"].max()
    corte = data_extracao - pd.DateOffset(months=mob_alvo)

    flag_no_alvo = inad.loc[inad["mob"] == mob_alvo].set_index("id_cliente")["flag_inadimplente"]
    base = conc.loc[conc["data_concessao"] <= corte].copy()
    base["mau"] = base["id_cliente"].map(flag_no_alvo).fillna(0).astype(int)
    return base


# % dos contratos inadimplentes em cada MOB.
# Usa só contratos com pelo menos mob_max meses de observação, e o denominador é sempre
# o total de contratos: quem quitou sai da base de inadimplência, mas continua contando como "não inadimplente".
def curva_de_safra(conc: pd.DataFrame, inad: pd.DataFrame, mob_max: int = 15) -> pd.Series:
    inad = adicionar_mob(inad, conc)
    corte = inad["data_referencia"].max() - pd.DateOffset(months=mob_max)
    coorte = conc.loc[conc["data_concessao"] <= corte, "id_cliente"]
    inad = inad[inad["id_cliente"].isin(coorte) & inad["mob"].between(1, mob_max)]
    return inad.groupby("mob")["flag_inadimplente"].sum() / len(coorte)

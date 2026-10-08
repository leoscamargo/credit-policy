# Scorecard: faixas, WoE, IV, regressão logística, escala de pontos e métricas.
import numpy as np
import pandas as pd

from scipy.stats import ks_2samp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

PDO, SCORE_BASE, ODDS_BASE = 50, 600, 20  # 600 pontos = 20 bons para 1 mau; +50 pontos dobra
FATOR = PDO / np.log(2)
OFFSET = SCORE_BASE - FATOR * np.log(ODDS_BASE)
ROTULOS_FAIXA = ["E", "D", "C", "B", "A"]  # do pior para o melhor score


# faixas e WoE
# Para cada faixa: total, maus, taxa de maus, WoE e parcela do IV.
# +0,5 evita divisão por zero em faixas sem maus (ou sem bons)
def _tabela_woe(faixa: pd.Series, y: pd.Series) -> pd.DataFrame:
    t = pd.DataFrame({"faixa": faixa, "mau": y}).groupby("faixa", observed=True)["mau"].agg(
        total="count", maus="sum")
    t["bons"] = t["total"] - t["maus"]
    t["taxa_maus"] = t["maus"] / t["total"]
    pct_bons = (t["bons"] + 0.5) / (t["bons"].sum() + 0.5 * len(t))
    pct_maus = (t["maus"] + 0.5) / (t["maus"].sum() + 0.5 * len(t))
    t["woe"] = np.log(pct_bons / pct_maus)
    t["iv"] = (pct_bons - pct_maus) * t["woe"]
    return t


def _e_monotonico(valores: np.ndarray) -> bool:
    d = np.diff(valores)
    return bool(np.all(d >= 0) or np.all(d <= 0))

# Aprende as faixas e o WoE de uma variável no TREINO e aplica em qualquer base.
class Agrupador:
    def __init__(self, variavel: str, categorica: bool = False, n_faixas: int = 5):
        self.variavel, self.categorica, self.n_faixas = variavel, categorica, n_faixas

    def fit(self, x: pd.Series, y: pd.Series):
        if self.categorica:
            self.cortes = None
            self.tabela = _tabela_woe(x.astype(str), y)
        else:
            # quantis do treino; valores repetidos (ex.: muitos zeros) reduzem o nº de faixas
            # garante WoE monotônico juntando faixas vizinhas quando a ordem quebra
            cortes = np.unique(np.quantile(x, np.linspace(0, 1, self.n_faixas + 1))[1:-1])
            while True:
                tabela = _tabela_woe(self._cortar(x, cortes), y)
                if _e_monotonico(tabela["woe"].values) or len(cortes) <= 1:
                    break
                diferencas = np.abs(np.diff(tabela["woe"].values))
                cortes = np.delete(cortes, int(np.argmin(diferencas)))
            self.cortes, self.tabela = cortes, tabela
        self.iv = float(self.tabela["iv"].sum())
        self.woe_padrao = 0.0  # categoria nunca vista no treino = neutra
        return self

    @staticmethod
    def _cortar(x: pd.Series, cortes: np.ndarray) -> pd.Series:
        limites = np.concatenate([[-np.inf], cortes, [np.inf]])
        return pd.cut(x, bins=limites, right=True)

    def faixas(self, x: pd.Series) -> pd.Series:
        return x.astype(str) if self.categorica else self._cortar(x, self.cortes)

    def transform(self, x: pd.Series) -> pd.Series:
        woe = self.faixas(x).map(self.tabela["woe"]).astype(float)
        return woe.fillna(self.woe_padrao)

# IV de cada variável, do mais forte para o mais fraco.
def calcular_iv(df: pd.DataFrame, variaveis: list, categoricas: list, alvo: str = "mau"):
    linhas = []
    for v in variaveis:
        ag = Agrupador(v, categorica=v in categoricas).fit(df[v], df[alvo])
        linhas.append({"variavel": v, "iv": ag.iv, "leitura": leitura_iv(ag.iv)})
    return pd.DataFrame(linhas).sort_values("iv", ascending=False).reset_index(drop=True)


def leitura_iv(iv: float) -> str:
    if iv < 0.02:
        return "Inútil"
    if iv < 0.1:
        return "Fraco"
    if iv < 0.3:
        return "Médio"
    if iv < 0.5:
        return "Forte"
    return "Muito forte (conferir)"


# Modelo
# Regressão logística sobre variáveis em WoE, com saída em pontos (0 a 1000).
class Scorecard:
    def __init__(self, variaveis: list, categoricas: list):
        self.variaveis, self.categoricas = variaveis, categoricas

    def fit(self, df: pd.DataFrame, alvo: str = "mau"):
        self.agrupadores = {v: Agrupador(v, v in self.categoricas).fit(df[v], df[alvo])
                            for v in self.variaveis}
        # C alto = praticamente sem regularização: os coeficientes refletem só os dados
        self.modelo = LogisticRegression(C=1e6, max_iter=1000)
        self.modelo.fit(self._woe(df), df[alvo])
        return self

    def _woe(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({v: self.agrupadores[v].transform(df[v]) for v in self.variaveis},
                            index=df.index)

    def prob_mau(self, df: pd.DataFrame) -> np.ndarray:
        return self.modelo.predict_proba(self._woe(df))[:, 1]

    def score(self, df: pd.DataFrame) -> np.ndarray:
        return prob_para_score(self.prob_mau(df))

    def coeficientes(self) -> pd.Series:
        return pd.Series(self.modelo.coef_[0], index=self.variaveis, name="coeficiente")

# Pontos de cada faixa de cada variável: a soma, mais a base é o score do cliente.
    def tabela_de_pontos(self) -> pd.DataFrame:
        n = len(self.variaveis)
        base = OFFSET - FATOR * self.modelo.intercept_[0]
        linhas = []
        for v, coef in self.coeficientes().items():
            t = self.agrupadores[v].tabela
            for faixa, linha in t.iterrows():
                linhas.append({"variavel": v, "faixa": str(faixa), "contratos": int(linha["total"]),
                                "taxa_maus": linha["taxa_maus"], "woe": linha["woe"],
                                "pontos": round(base / n - FATOR * coef * linha["woe"])})
        return pd.DataFrame(linhas)


def prob_para_score(prob_mau: np.ndarray) -> np.ndarray:
    prob_mau = np.clip(prob_mau, 1e-6, 1 - 1e-6)
    score = OFFSET + FATOR * np.log((1 - prob_mau) / prob_mau)
    return np.clip(np.round(score), 0, 1000)


# Métricas
# AUC, Gini e KS. Use maior_e_pior=True quando passar probabilidade de mau.
def metricas(y, score_ou_prob, maior_e_pior: bool = False) -> dict:
    y = np.asarray(y)
    risco = np.asarray(score_ou_prob) if maior_e_pior else -np.asarray(score_ou_prob)
    auc = roc_auc_score(y, risco)
    ks = ks_2samp(risco[y == 1], risco[y == 0]).statistic
    return {"auc": round(float(auc), 3), "gini": round(float(2 * auc - 1), 3),
            "ks": round(float(ks), 3)}

# Cortes das faixas de risco = quantis do score no treino.
def cortes_das_faixas(score_treino: np.ndarray, n: int = 5) -> np.ndarray:
    return np.quantile(score_treino, np.linspace(0, 1, n + 1))[1:-1].round()


def atribuir_faixa(score: np.ndarray, cortes: np.ndarray) -> pd.Categorical:
    limites = np.concatenate([[-np.inf], cortes, [np.inf]])
    return pd.cut(score, bins=limites, labels=ROTULOS_FAIXA, right=False)


# Population Stability Index entre duas distribuições de score, nas mesmas faixas."""
def psi(esperado, atual, cortes: np.ndarray) -> float:
    limites = np.concatenate([[-np.inf], cortes, [np.inf]])
    e = np.histogram(esperado, limites)[0] / len(esperado)
    a = np.histogram(atual, limites)[0] / len(atual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))

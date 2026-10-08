# Gráficos do relatório. Cada função devolve a figura e, se pedido, salva em figuras/.
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

AZUL, LARANJA, CINZA, TEXTO = "#2a78d6", "#eb6834", "#8a8a86", "#1f2933"
FIG_DIR = Path(__file__).resolve().parents[1] / "figuras"

plt.rcParams.update({
    "figure.dpi": 110, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#c9c9c4", "axes.labelcolor": TEXTO, "xtick.color": TEXTO,
    "ytick.color": TEXTO, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def _finalizar(fig, nome):
    fig.tight_layout()
    if nome:
        FIG_DIR.mkdir(exist_ok=True)
        fig.savefig(FIG_DIR / f"{nome}.png", dpi=200, bbox_inches="tight", facecolor="white")
    return fig


def curva_safra(curva: pd.Series, mob_alvo: int = 12, nome: str | None = None):
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.plot(curva.index, curva.values * 100, color=AZUL, lw=2, marker="o", ms=5)
    ax.axvline(mob_alvo, color=CINZA, ls="--", lw=1)
    ax.annotate(f"MOB {mob_alvo}: {curva[mob_alvo]:.1%}".replace(".", ","),
                (mob_alvo, curva[mob_alvo] * 100), xytext=(8, -28), textcoords="offset points")
    ax.set_title("A inadimplência cresce até o 12º mês e estabiliza")
    ax.set_xlabel("Meses desde a concessão (MOB)")
    ax.set_ylabel("% dos contratos inadimplentes")
    ax.set_xticks(curva.index)
    ax.set_ylim(0, None)
    ax.grid(axis="y", color="#e6e6e2", lw=0.8)
    return _finalizar(fig, nome)


def degrau_comprometimento(df: pd.DataFrame, degrau: float = 0.30, nome: str | None = None):
    limites = np.arange(0.10, 0.95, 0.05)
    faixa = pd.cut(df["comprometimento"].clip(upper=0.8999), limites, right=False)
    taxa = df.groupby(faixa, observed=True)["mau"].mean() * 100
    centros = [i.left * 100 + 2.5 for i in taxa.index]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.bar(centros, taxa.values, width=4.2, color=AZUL)
    ax.axvline(degrau * 100, color=TEXTO, ls="--", lw=1)
    ax.annotate(f"{degrau:.0%} da renda", (degrau * 100, taxa.max()), xytext=(6, -2),
                textcoords="offset points")
    ax.set_title("O risco salta quando a parcela passa de 30% da renda")
    ax.set_xlabel("Parcela / renda (%)")
    ax.set_ylabel("% maus pagadores (MOB 12)")
    ax.grid(axis="y", color="#e6e6e2", lw=0.8)
    ax.set_axisbelow(True)
    return _finalizar(fig, nome)


def maus_por_faixa(treino: pd.DataFrame, teste: pd.DataFrame, nome: str | None = None):
    ordem = ["A", "B", "C", "D", "E"]
    tr = treino.groupby("faixa", observed=False)["mau"].mean().reindex(ordem) * 100
    te = teste.groupby("faixa", observed=False)["mau"].mean().reindex(ordem) * 100
    x = np.arange(len(ordem))
    fig, ax = plt.subplots(figsize=(8, 3.8))
    b1 = ax.bar(x - 0.2, tr.values, 0.38, color=AZUL, label="Treino (ago/2019–dez/2020)")
    b2 = ax.bar(x + 0.2, te.values, 0.38, color=LARANJA,
                label="Teste fora do tempo (jan–jul/2021)")
    for barras in (b1, b2):
        ax.bar_label(barras, labels=[f"{v:.1f}%".replace(".", ",") for v in barras.datavalues],
                        padding=2, fontsize=8.5)
    ax.set_xticks(x, [f"Faixa {f}" for f in ordem])
    ax.set_title("As faixas ordenam o risco e se repetem no período de teste")
    ax.set_ylabel("% maus pagadores")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="y", color="#e6e6e2", lw=0.8)
    ax.set_axisbelow(True)
    return _finalizar(fig, nome)


# Azul = lucro, vermelho = prejuízo, cinza claro = perto de zero.
def mapa_margem(matriz_margem: pd.DataFrame, nome: str | None = None):
    cores = LinearSegmentedColormap.from_list("margem", ["#d9534a", "#f0efec", "#2a78d6"])
    valores = matriz_margem.values * 100
    limite = max(abs(np.nanmin(valores)), abs(np.nanmax(valores)))
    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    ax.imshow(valores, cmap=cores, norm=TwoSlopeNorm(0, -limite, limite), aspect="auto")
    for i in range(valores.shape[0]):
        for j in range(valores.shape[1]):
            v = valores[i, j]
            ax.text(j, i, f"{v:+.1f}%".replace(".", ","), ha="center", va="center",
                    color="white" if abs(v) > 0.6 * limite else TEXTO, fontsize=9)
    ax.set_xticks(range(valores.shape[1]), matriz_margem.columns)
    ax.set_yticks(range(valores.shape[0]), [f"Faixa {f}" for f in matriz_margem.index])
    ax.set_xlabel("Parcela / renda")
    ax.set_title("Margem sobre o valor emprestado: azul = lucro, vermelho = prejuízo")
    for lado in ax.spines.values():
        lado.set_visible(False)
    ax.tick_params(length=0)
    return _finalizar(fig, nome)

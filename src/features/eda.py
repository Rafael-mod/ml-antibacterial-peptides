"""
src/reports/eda.py

Análise Exploratória de Dados (EDA) e Visualização:
1. Carrega o 'dataset_treino.csv' gerado na etapa de engenharia de features.
2. Analisa o balanceamento de classes e estatísticas descritivas.
3. Avalia distribuições e correlações de propriedades clássicas, Z-scales, PC6 e PseAAC.
4. Calcula o poder discriminativo (AUC univariado via Mann-Whitney U) para todas as métricas.
5. Analisa a composição global de aminoácidos (AAC) e o perfil posicional (One-Hot).
6. Gera projeções multivariadas (PCA e t-SNE) com o espaço expandido.
7. Salva todos os gráficos gerados em 'reports/figuras_eda/'.

Uso:
    python src/reports/eda.py
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# Estilo global dos gráficos
sns.set_theme(style="whitegrid", context="notebook")

# ==========================================
# Configuração de Diretórios do Projeto
# ==========================================
ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"
PASTA_FIG = REPORTS_DIR / "figuras_eda"

INPUT_CSV = PROC / "dataset_treino.csv"

# ==========================================
# Constantes do Dataset
# ==========================================
AMINOACIDOS = list("ACDEFGHIKLMNPQRSTVWY")
NOMES = {0: "Negativo", 1: "Positivo"}
PALETA = {"Negativo": "#4C72B0", "Positivo": "#DD8452"}


# ==========================================
# Utilitários de Mídia e Carga de Dados
# ==========================================
def salvar(nome: str, fig: plt.Figure = None) -> None:
    PASTA_FIG.mkdir(parents=True, exist_ok=True)
    fig = fig or plt.gcf()
    caminho_saida = PASTA_FIG / nome
    fig.savefig(caminho_saida, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Figura salva: {caminho_saida.relative_to(ROOT)}")


def carregar_dados() -> tuple[pd.DataFrame, dict[str, list[str]]]:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {INPUT_CSV}. "
            "Execute 'python src/features/build_features.py' primeiro."
        )
    
    df = pd.read_csv(INPUT_CSV)
    df["classe"] = df["label"].map(NOMES)
    
    # Categorização das colunas
    grupos = {
        "aac": [c for c in df.columns if c.startswith("AAC_")],
        "oh": [c for c in df.columns if c.startswith("OH_")],
        "zscale": [c for c in df.columns if c.startswith("Zscale_")],
        "pc6": [c for c in df.columns if c.startswith("PC6_")],
        "pseaac": [c for c in df.columns if c.startswith("PseAAC_")]
    }
    
    prefixos_conhecidos = ("AAC_", "OH_", "Zscale_", "PC6_", "PseAAC_")
    excluidas = {"id", "sequence", "label", "classe"}
    grupos["fis"] = [c for c in df.columns if c not in excluidas and not c.startswith(prefixos_conhecidos)]
    
    return df, grupos


def grade(n: int, ncols: int = 4, w: float = 4.2, h: float = 3.2) -> tuple[plt.Figure, np.ndarray]:
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * w, nrows * h))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[n:]:
        ax.axis("off")
    return fig, axes


# ==========================================
# 1) Visão Geral do Dataset
# ==========================================
def resumo_geral(df: pd.DataFrame, grupos: dict[str, list[str]]) -> None:
    print("\n=== [1/7] Visão Geral do Dataset ===")
    print(f"Shape Total: {df.shape}")
    print(f"Features: Físico-químicas ({len(grupos['fis'])}) | AAC ({len(grupos['aac'])}) | "
          f"Z-scale ({len(grupos['zscale'])}) | PC6 ({len(grupos['pc6'])}) | "
          f"PseAAC ({len(grupos['pseaac'])}) | One-Hot ({len(grupos['oh'])})")
    
    print("\nBalanceamento de Classes:")
    print(df["classe"].value_counts().to_string())
    print(f"\nTotal de Valores Nulos: {df.isna().sum().sum()}")
    print(f"Sequências Duplicadas: {df['sequence'].duplicated().sum()}")


def plot_balanceamento(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(4, 3.5))
    sns.countplot(data=df, x="classe", hue="classe", palette=PALETA, legend=False, ax=ax)
    for p in ax.patches:
        ax.annotate(int(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom")
    ax.set_title("Balanceamento das Classes")
    ax.set_xlabel("")
    salvar("01_balanceamento.png", fig)


# ==========================================
# 2) Análise de Distribuições (Numéricas)
# ==========================================
def plot_distribuicoes(df: pd.DataFrame, cols: list[str], titulo: str, arquivo: str) -> None:
    fig, axes = grade(len(cols))
    for ax, col in zip(axes, cols):
        sns.histplot(data=df, x=col, hue="classe", palette=PALETA, stat="density",
                     common_norm=False, kde=True, bins=30, alpha=0.4, ax=ax, legend=(col == cols[0]))
        ax.set_title(col)
        ax.set_xlabel("")
    fig.suptitle(titulo, y=1.02, fontsize=14)
    salvar(arquivo, fig)


def plot_boxplots(df: pd.DataFrame, cols: list[str], titulo: str, arquivo: str) -> None:
    fig, axes = grade(len(cols))
    for ax, col in zip(axes, cols):
        sns.boxplot(data=df, x="classe", y=col, hue="classe", palette=PALETA,
                    legend=False, fliersize=2, ax=ax)
        ax.set_title(col)
        ax.set_xlabel("")
        ax.set_ylabel("")
    fig.suptitle(titulo, y=1.02, fontsize=14)
    salvar(arquivo, fig)


# ==========================================
# 3) Poder Discriminativo Univariado
# ==========================================
def poder_discriminativo(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    pos, neg = df[df["label"] == 1], df[df["label"] == 0]
    linhas = []
    for c in cols:
        u, p = mannwhitneyu(pos[c], neg[c], alternative="two-sided")
        auc = u / (len(pos) * len(neg))
        linhas.append({"feature": c, "auc": auc, "delta_auc": auc - 0.5, "p_valor": p})
    return (pd.DataFrame(linhas)
            .sort_values("delta_auc", key=abs, ascending=False).reset_index(drop=True))


def plot_poder_discriminativo(tabela: pd.DataFrame, top: int = 30) -> None:
    t = tabela.head(top).iloc[::-1]
    cores = np.where(t["delta_auc"] > 0, PALETA["Positivo"], PALETA["Negativo"])
    fig, ax = plt.subplots(figsize=(8, 0.35 * len(t) + 1))
    ax.barh(t["feature"], t["delta_auc"], color=cores)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("AUC − 0,5   (Laranja: Maior nos Positivos | Azul: Maior nos Negativos)")
    ax.set_title(f"Top {len(t)} Features (Poder Discriminativo Isolado)")
    salvar("05_poder_discriminativo_global.png", fig)


# ==========================================
# 4) Composição de Aminoácidos (AAC)
# ==========================================
def plot_aac(df: pd.DataFrame, aac: list[str]) -> None:
    medias = df.groupby("classe")[aac].mean().T
    medias.index = [c.replace("AAC_", "") for c in medias.index]
    ordem = medias.sort_values("Positivo", ascending=False).index

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    medias.loc[ordem, ["Negativo", "Positivo"]].plot.bar(
        ax=axes[0], color=[PALETA["Negativo"], PALETA["Positivo"]], width=0.8)
    axes[0].set_title("Composição Média de Aminoácidos")
    axes[0].set_ylabel("Fração Média")
    axes[0].tick_params(axis="x", rotation=0)

    lfc = np.log2((medias["Positivo"] + 1e-9) / (medias["Negativo"] + 1e-9)).sort_values()
    axes[1].bar(lfc.index, lfc.values, color=np.where(lfc > 0, PALETA["Positivo"], PALETA["Negativo"]))
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set_title("log2(Positivo / Negativo): Enriquecimento por Aminoácido")
    axes[1].set_ylabel("log2 Fold Change")
    salvar("06_aac.png", fig)


# ==========================================
# 5) Perfil Posicional (One-Hot)
# ==========================================
def plot_one_hot_posicional(df: pd.DataFrame, oh: list[str], min_seqs: int = 30) -> None:
    n_pos = len(oh) // len(AMINOACIDOS)
    comp = df["sequence"].str.len().values
    matriz = df[oh].values.reshape(len(df), n_pos, len(AMINOACIDOS))
    posicoes = np.arange(1, n_pos + 1)

    freq = {}
    for rotulo in (1, 0):
        m = (df["label"] == rotulo).values
        base = (comp[m][:, None] >= posicoes).sum(axis=0)
        freq[rotulo] = np.divide(matriz[m].sum(axis=0), base[:, None],
                                 out=np.zeros((n_pos, len(AMINOACIDOS))), where=base[:, None] > 0)
        freq[f"base{rotulo}"] = base

    validas = (freq["base1"] >= min_seqs) & (freq["base0"] >= min_seqs)
    ate = int(np.where(validas)[0].max()) + 1 if np.any(validas) else n_pos
    rot_x = posicoes[:ate]
    dif = (freq[1] - freq[0])[:ate].T

    fig, axes = plt.subplots(3, 1, figsize=(14, 13), sharex=True)
    for ax, (titulo, dados, cmap, kw) in zip(axes, [
        ("Positivos: Frequência", freq[1][:ate].T, "viridis", {}),
        ("Negativos: Frequência", freq[0][:ate].T, "viridis", {}),
        ("Diferença (Positivo − Negativo)", dif, "coolwarm", {"center": 0, "vmin": -np.abs(dif).max(), "vmax": np.abs(dif).max()}),
    ]):
        sns.heatmap(dados, ax=ax, cmap=cmap, yticklabels=AMINOACIDOS, xticklabels=rot_x, cbar_kws={"shrink": 0.8}, **kw)
        ax.set_title(titulo)
        ax.set_ylabel("Aminoácido")
        ax.tick_params(axis="y", rotation=0, labelsize=8)
    axes[-1].set_xlabel(f"Posição (N-terminal → C-terminal); Posições com ≥ {min_seqs} sequências")
    salvar("08_one_hot_posicional.png", fig)


# ==========================================
# 6) Projeções Multivariadas (PCA / t-SNE)
# ==========================================
def plot_projecoes(df: pd.DataFrame, cols: list[str], seed: int = 42) -> None:
    X = StandardScaler().fit_transform(df[cols])
    
    print("\n  -> Computando PCA...")
    pca = PCA(n_components=2, random_state=seed)
    Z = pca.fit_transform(X)
    var = pca.explained_variance_ratio_ * 100

    print("  -> Computando t-SNE (Isso pode levar alguns segundos)...")
    tsne = TSNE(n_components=2, perplexity=30, init="pca", random_state=seed)
    T = tsne.fit_transform(X)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, dados, titulo, rot in [
        (axes[0], Z, "PCA (Múltiplas Famílias de Features)", (f"PC1 ({var[0]:.1f}%)", f"PC2 ({var[1]:.1f}%)")),
        (axes[1], T, "t-SNE (Múltiplas Famílias de Features)", ("Dimensão 1", "Dimensão 2")),
    ]:
        sns.scatterplot(x=dados[:, 0], y=dados[:, 1], hue=df["classe"], palette=PALETA,
                        s=14, alpha=0.6, linewidth=0, ax=ax)
        ax.set_title(titulo)
        ax.set_xlabel(rot[0])
        ax.set_ylabel(rot[1])
        ax.legend(title="")
    salvar("09_projecoes_pca_tsne.png", fig)


# ==========================================
# Execução Principal
# ==========================================
def main():
    print("=== Iniciando Análise Exploratória de Dados (EDA) ===")
    df, g = carregar_dados()

    resumo_geral(df, g)
    
    print("\n=== [2/7] Gerando Gráficos de Distribuição ===")
    plot_balanceamento(df)
    
    # Físico-químicas
    plot_distribuicoes(df, g["fis"], "Propriedades Físico-Químicas Clássicas", "02a_distribuicoes_fis.png")
    plot_boxplots(df, g["fis"], "Boxplots: Físico-Químicas Clássicas", "03a_boxplots_fis.png")
    
    # Z-scales e PC6
    plot_distribuicoes(df, g["zscale"] + g["pc6"], "Propriedades Latentes (Z-scale e PC6)", "02b_distribuicoes_latentes.png")
    
    # PseAAC
    if g["pseaac"]:
        plot_distribuicoes(df, g["pseaac"], "Correlações de Sequência (PseAAC)", "02c_distribuicoes_pseaac.png")

    print("\n=== [3/7] Calculando Poder Discriminativo (Todas as Numéricas) ===")
    todas_numericas = g["fis"] + g["aac"] + g["zscale"] + g["pc6"] + g["pseaac"]
    tabela_auc = poder_discriminativo(df, todas_numericas)
    plot_poder_discriminativo(tabela_auc, top=30)
    
    print("\nTop 15 Features Globais (AUC):")
    print(tabela_auc.head(15).round(4).to_string(index=False))

    print("\n=== [4/7] Analisando Composição de Aminoácidos (AAC) ===")
    plot_aac(df, g["aac"])

    print("\n=== [5/7] Gerando Perfis Posicionais (One-Hot) ===")
    plot_one_hot_posicional(df, g["oh"])

    print("\n=== [6/7] Gerando Projeções Multivariadas ===")
    plot_projecoes(df, todas_numericas)

    print(f"\n=== [7/7] Sucesso! Figuras salvas em:\n -> {PASTA_FIG.resolve()}")


if __name__ == "__main__":
    main()
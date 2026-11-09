"""
src/reports/eda.py

Análise Exploratória de Dados (EDA) Avançada do Espaço Filtrado:
1. Carrega 'X_train_limpo.csv' e 'y_train.csv' (gerados pós-RFECV/Seleção).
2. Calcula o poder discriminativo (Mann-Whitney U) das features sobreviventes.
3. Plota Distribuições e Boxplots exclusivas do Top 12 melhores descritores.
4. Analisa AAC e One-Hot (somente se essas features tiverem sobrevivido ao filtro).
5. Gera Matriz de Correlação das Top 20 features e Projeções (PCA/t-SNE).

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

# Estilo global
sns.set_theme(style="whitegrid", context="paper")

# ==========================================
# Configuração de Diretórios do Projeto
# ==========================================
ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"
PASTA_FIG = REPORTS_DIR / "figuras_eda"

X_CSV = PROC / "X_train_limpo.csv"
Y_CSV = PROC / "y_train.csv"

# ==========================================
# Constantes do Dataset
# ==========================================
NOMES = {0: "Negativo", 1: "Positivo"}
PALETA = {"Negativo": "#4C72B0", "Positivo": "#DD8452"}

# ==========================================
# Utilitários
# ==========================================
def salvar(nome: str, fig: plt.Figure = None) -> None:
    PASTA_FIG.mkdir(parents=True, exist_ok=True)
    fig = fig or plt.gcf()
    caminho_saida = PASTA_FIG / nome
    fig.savefig(caminho_saida, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Figura salva: {caminho_saida.relative_to(ROOT)}")


def carregar_dados() -> tuple[pd.DataFrame, dict[str, list[str]]]:
    if not X_CSV.exists() or not Y_CSV.exists():
        raise FileNotFoundError(
            f"Arquivos limpos não encontrados em {PROC}. "
            "Execute a seleção de atributos (tratamentoColunas.py) primeiro."
        )
    
    X = pd.read_csv(X_CSV)
    y = pd.read_csv(Y_CSV).squeeze()
    
    df = X.copy()
    df["label"] = y.values
    df["classe"] = df["label"].map(NOMES)
    
    # Categorização dinâmica baseada apenas nas colunas que sobreviveram ao funil
    grupos = {
        "aac": [c for c in X.columns if c.startswith("AAC_")],
        "oh": [c for c in X.columns if c.startswith("OH_")],
        "numericas": [c for c in X.columns if not c.startswith("OH_")]
    }
    return df, grupos


# ==========================================
# Análises e Gráficos
# ==========================================
def poder_discriminativo(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    print("\n  -> Calculando Teste de Mann-Whitney U para todas as variáveis...")
    pos, neg = df[df["label"] == 1], df[df["label"] == 0]
    linhas = []
    
    for c in cols:
        u, p = mannwhitneyu(pos[c], neg[c], alternative="two-sided")
        auc = u / (len(pos) * len(neg))
        linhas.append({"feature": c, "auc": auc, "delta_auc": auc - 0.5, "p_valor": p})
        
    return pd.DataFrame(linhas).sort_values("delta_auc", key=abs, ascending=False).reset_index(drop=True)


def plot_top_features_isoladas(df: pd.DataFrame, tabela_auc: pd.DataFrame, top_k: int = 12) -> None:
    top_cols = tabela_auc["feature"].head(top_k).tolist()
    
    # Boxplots
    fig_box, axes_box = plt.subplots(3, 4, figsize=(14, 10))
    axes_box = axes_box.ravel()
    
    for ax, col in zip(axes_box, top_cols):
        sns.boxplot(data=df, x="classe", y=col, hue="classe", palette=PALETA, legend=False, fliersize=2, ax=ax)
        ax.set_title(col[:25], fontsize=10)
        ax.set(xlabel="", ylabel="")
    fig_box.suptitle(f"Top {top_k} Descritores - Boxplots", y=1.02, fontsize=14)
    fig_box.tight_layout()
    salvar("01_top12_boxplots.png", fig_box)

    # Densidades
    fig_kde, axes_kde = plt.subplots(3, 4, figsize=(14, 10))
    axes_kde = axes_kde.ravel()
    
    for ax, col in zip(axes_kde, top_cols):
        sns.kdeplot(data=df, x=col, hue="classe", palette=PALETA, fill=True, common_norm=False, alpha=0.4, ax=ax, legend=(col == top_cols[0]))
        ax.set_title(col[:25], fontsize=10)
        ax.set(xlabel="", ylabel="")
    fig_kde.suptitle(f"Top {top_k} Descritores - Distribuição de Densidade", y=1.02, fontsize=14)
    fig_kde.tight_layout()
    salvar("02_top12_densidades.png", fig_kde)


def plot_correlacao_top(df: pd.DataFrame, tabela_auc: pd.DataFrame, top_k: int = 20) -> None:
    top_cols = tabela_auc["feature"].head(top_k).tolist()
    corr = df[top_cols].corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0, vmin=-1, vmax=1, 
                annot=True, fmt=".1f", square=True, linewidths=.5, 
                cbar_kws={"shrink": .8}, annot_kws={"size": 7}, ax=ax)
    ax.set_title(f"Matriz de Correlação - Top {top_k} Features")
    salvar("03_correlacao_top20.png", fig)


def plot_poder_discriminativo(tabela: pd.DataFrame, top: int = 30) -> None:
    t = tabela.head(top).iloc[::-1]
    cores = np.where(t["delta_auc"] > 0, PALETA["Positivo"], PALETA["Negativo"])
    fig, ax = plt.subplots(figsize=(8, 0.35 * len(t) + 1))
    ax.barh(t["feature"], t["delta_auc"], color=cores)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("AUC − 0,5   (Laranja: Maior nos Positivos | Azul: Maior nos Negativos)")
    ax.set_title(f"Top {len(t)} Features (Poder Discriminativo Isolado)")
    salvar("04_poder_discriminativo_rank.png", fig)


def plot_projecoes(df: pd.DataFrame, cols: list[str], seed: int = 42) -> None:
    X = StandardScaler().fit_transform(df[cols])
    
    print("  -> Computando PCA...")
    pca = PCA(n_components=2, random_state=seed)
    Z = pca.fit_transform(X)
    var = pca.explained_variance_ratio_ * 100

    print("  -> Computando t-SNE (Pode levar alguns segundos)...")
    tsne = TSNE(n_components=2, perplexity=30, init="pca", random_state=seed)
    T = tsne.fit_transform(X)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, dados, titulo, rot in [
        (axes[0], Z, "PCA (Espaço Filtrado)", (f"PC1 ({var[0]:.1f}%)", f"PC2 ({var[1]:.1f}%)")),
        (axes[1], T, "t-SNE (Espaço Filtrado)", ("Dimensão 1", "Dimensão 2")),
    ]:
        sns.scatterplot(x=dados[:, 0], y=dados[:, 1], hue=df["classe"], palette=PALETA,
                        s=15, alpha=0.6, linewidth=0, ax=ax)
        ax.set_title(titulo)
        ax.set(xlabel=rot[0], ylabel=rot[1])
        if ax == axes[0]: ax.legend(title="")
        else: ax.get_legend().remove()
    salvar("05_projecoes_multivariadas.png", fig)


def plot_aac_sobreviventes(df: pd.DataFrame, aac_cols: list[str]) -> None:
    if not aac_cols:
        print("  -> Aviso: Nenhuma feature AAC sobreviveu ao filtro. Pulando plotagem.")
        return
        
    medias = df.groupby("classe")[aac_cols].mean().T
    medias.index = [c.replace("AAC_", "") for c in medias.index]
    ordem = medias.sort_values("Positivo", ascending=False).index

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    medias.loc[ordem, ["Negativo", "Positivo"]].plot.bar(
        ax=axes[0], color=[PALETA["Negativo"], PALETA["Positivo"]], width=0.8)
    axes[0].set_title("AAC (Apenas os que passaram no filtro)")
    axes[0].set_ylabel("Fração Média")
    axes[0].tick_params(axis="x", rotation=0)

    lfc = np.log2((medias["Positivo"] + 1e-9) / (medias["Negativo"] + 1e-9)).sort_values()
    axes[1].bar(lfc.index, lfc.values, color=np.where(lfc > 0, PALETA["Positivo"], PALETA["Negativo"]))
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set_title("log2(Positivo / Negativo): Enriquecimento")
    salvar("06_aac_sobreviventes.png", fig)


# ==========================================
# Execução Principal
# ==========================================
def main():
    print("=== EDA: Espaço de Features Filtradas (Pós-RFECV) ===")
    df, g = carregar_dados()

    print(f"Shape: {df.shape[0]} amostras | {len(g['numericas']) + len(g['oh'])} descritores limpos")
    
    tabela_auc = poder_discriminativo(df, g["numericas"])
    
    print("\n[1/4] Gerando Análise Isolada do Top 12 Descritores...")
    plot_top_features_isoladas(df, tabela_auc, top_k=12)
    
    print("[2/4] Gerando Matriz de Correlação (Top 20)...")
    plot_correlacao_top(df, tabela_auc, top_k=20)
    
    print("[3/4] Gerando Gráfico de Rank (Poder Discriminativo)...")
    plot_poder_discriminativo(tabela_auc, top=30)
    
    print("[4/4] Gerando Projeções Multivariadas (PCA/t-SNE) de todas as features limpas...")
    plot_projecoes(df, g["numericas"] + g["oh"])
    
    # Análises Biológicas Específicas
    plot_aac_sobreviventes(df, g["aac"])

    print(f"\n=== Sucesso! Figuras salvas em:\n -> {PASTA_FIG.resolve()}")


if __name__ == "__main__":
    main()
"""
src/models/shap_analysis.py

Análise de explicabilidade (SHAP) para o modelo campeão (ex: SVM RBF calibrado).
- Compatível com CalibratedClassifierCV + Pipeline(PreprocessadorDinamico).
- Evita embaralhamento de colunas do ColumnTransformer.
- Usa peptídeos reais (medoides) no background para preservar colunas binárias (OHE).
- Paraleliza o KernelExplainer em blocos e audita o peso por família de features.
"""

import sys
import time
import warnings
from pathlib import Path
from typing import List, Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.preprocessing import StandardScaler

# Garante importação do PreprocessadorDinamico ao carregar o .joblib
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

sns.set_theme(style="whitegrid", context="notebook")

PROC_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figuras_shap"
TAB_DIR = REPORTS_DIR / "tabelas"

SEED = 42


# ==========================================
# 1. Utilitários de Background e Famílias
# ==========================================
def selecionar_background_medoides(
    X_train: pd.DataFrame, n_medoides: int = 50, seed: int = SEED
) -> pd.DataFrame:
    """
    Agrupa o treino no espaço padronizado e escolhe o peptídeo real mais próximo
    de cada centroide. Mantém variáveis binárias (OHE_N/OHE_C) estritamente em 0/1.
    """
    X_scaled = StandardScaler().fit_transform(X_train)
    km = KMeans(n_clusters=n_medoides, random_state=seed, n_init=10).fit(X_scaled)
    idx_medoides = pairwise_distances_argmin(km.cluster_centers_, X_scaled)
    return X_train.iloc[idx_medoides].reset_index(drop=True)


def classificar_familia(col: str) -> str:
    """Mapeia cada nome de coluna para sua família de descritores."""
    c = str(col)
    if c.startswith(("OHE_N", "OH_N")):
        return "OHE_N"
    if c.startswith(("OHE_C", "OH_C")):
        return "OHE_C"
    if c.startswith("AAC_"):
        return "AAC"
    if c.startswith("PC6"):
        return "PC6"
    if c.startswith("SOC") or "Eisenberg" in c:
        return "SOC_Eisenberg"
    if c.startswith(("Z_", "Z1", "Z2", "Z3", "Z4", "Z5", "ZScale")) or "zscale" in c.lower():
        return "Z-scales"
    return "Fisico-quimica"


def _extrair_classe_positiva(shap_obj) -> np.ndarray:
    """Padroniza saída do SHAP entre versões antigas (list) e recentes (array 2D/3D)."""
    if isinstance(shap_obj, list):
        return np.asarray(shap_obj[1], dtype=float)
    arr = np.asarray(shap_obj, dtype=float)
    if arr.ndim == 3:
        return arr[:, :, 1]
    return arr


def _calcular_bloco_shap(
    modelo,
    background_df: pd.DataFrame,
    bloco_df: pd.DataFrame,
    feature_names: List[str],
    nsamples: int,
) -> np.ndarray:
    """Função top-level (picklável) para paralelizar o KernelExplainer por blocos."""
    def predict_fn(x: np.ndarray) -> np.ndarray:
        df_in = pd.DataFrame(x, columns=feature_names)
        if hasattr(modelo, "predict_proba"):
            return modelo.predict_proba(df_in)[:, 1]
        return modelo.decision_function(df_in)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        explainer = shap.KernelExplainer(predict_fn, background_df)
        sv = explainer.shap_values(bloco_df, nsamples=nsamples, silent=True)
    return _extrair_classe_positiva(sv)


# ==========================================
# 2. Gráficos
# ==========================================
def plot_familias(df_fam: pd.DataFrame, caminho_fig: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    sns.barplot(
        data=df_fam, y="familia", x="pct_impacto_total", ax=axes[0], color="#4c72b0"
    )
    axes[0].set(
        title="Participação no Impacto Total SHAP (%)",
        xlabel="% da Soma de |SHAP|",
        ylabel="",
    )
    for container in axes[0].containers:
        axes[0].bar_label(container, fmt="%.1f%%", padding=3)

    sns.barplot(
        data=df_fam, y="familia", x="impacto_medio_por_feature", ax=axes[1], color="#dd8452"
    )
    axes[1].set(
        title="Impacto Médio por Feature da Família",
        xlabel="Média de |SHAP| por atributo",
        ylabel="",
    )

    fig.suptitle("Auditoria SHAP por Família de Descritores", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(caminho_fig, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ==========================================
# 3. Pipeline Principal
# ==========================================
def executar_analise_shap(
    n_medoides: int = 50,
    nsamples: int = 200,
    max_amostras_teste: Optional[int] = None,
    n_jobs: int = -1,
) -> None:
    """
    Parâmetros:
      n_medoides: tamanho do background de peptídeos reais (50 é um ótimo equilíbrio).
      nsamples: número de perturbações por amostra no KernelExplainer (200-300).
      max_amostras_teste: se None, explica todo o X_test; defina ex. 300 para rodar mais rápido.
      n_jobs: núcleos de CPU usados para paralelizar os blocos.
    """
    print("=== Iniciando Análise SHAP (Modelo Campeão) ===")

    caminho_treino = PROC_DIR / "X_train_limpo.csv"
    caminho_teste = PROC_DIR / "X_test_limpo.csv"
    if not caminho_treino.exists() or not caminho_teste.exists():
        raise FileNotFoundError("Matrizes 'X_train_limpo.csv' ou 'X_test_limpo.csv' não encontradas.")

    X_train = pd.read_csv(caminho_treino)
    X_test = pd.read_csv(caminho_teste)
    feature_names = list(X_test.columns)

    # Subamostragem opcional do teste (estratificada se meta_test existir)
    if max_amostras_teste is not None and max_amostras_teste < len(X_test):
        meta_path = PROC_DIR / "meta_test.csv"
        if meta_path.exists():
            y_te = pd.read_csv(meta_path)["label"].astype(int)
            X_explicar = (
                X_test.assign(_y=y_te)
                .groupby("_y", group_keys=False)
                .apply(lambda g: g.sample(n=max_amostras_teste // 2, random_state=SEED))
                .drop(columns="_y")
                .reset_index(drop=True)
            )
        else:
            X_explicar = X_test.sample(n=max_amostras_teste, random_state=SEED).reset_index(drop=True)
    else:
        X_explicar = X_test.copy()

    # 1. Carregar modelo
    caminho_modelo = MODELS_DIR / "modelo_final_otimizado.joblib"
    if not caminho_modelo.exists():
        caminho_modelo = MODELS_DIR / "modelo_final.joblib"
    modelo = joblib.load(caminho_modelo)
    print(f"  -> Modelo carregado: {caminho_modelo.name} ({type(modelo).__name__})")
    print(f"  -> Explicando {len(X_explicar)} amostras x {len(feature_names)} features")

    # 2. Background com peptídeos reais (medoides)
    print(f"  -> Selecionando {n_medoides} peptídeos medoides de background...")
    background_df = selecionar_background_medoides(X_train, n_medoides=n_medoides, seed=SEED)

    # 3. Cálculo paralelo dos valores SHAP em blocos
    t0 = time.time()
    n_blocos = min(len(X_explicar), max(1, (joblib.cpu_count() or 4) * 2))
    indices = np.array_split(np.arange(len(X_explicar)), n_blocos)
    blocos = [X_explicar.iloc[idx] for idx in indices if len(idx) > 0]

    print(f"  -> Calculando KernelSHAP em {len(blocos)} blocos (nsamples={nsamples}, n_jobs={n_jobs})...")
    resultados = Parallel(n_jobs=n_jobs)(
        delayed(_calcular_bloco_shap)(modelo, background_df, b, feature_names, nsamples)
        for b in blocos
    )
    shap_vals = np.vstack(resultados)
    print(f"  -> SHAP concluído em {time.time() - t0:.1f}s | shape={shap_vals.shape}")

    # 4. Salvar artefatos brutos e tabelas
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        TAB_DIR / "shap_values_teste.npz",
        shap_values=shap_vals,
        feature_names=np.array(feature_names, dtype=object),
    )

    mean_abs = np.abs(shap_vals).mean(axis=0)
    total_abs = mean_abs.sum()

    corrs = []
    for i, col in enumerate(feature_names):
        if X_explicar[col].nunique() <= 1 or np.std(shap_vals[:, i]) == 0:
            corrs.append(0.0)
        else:
            corrs.append(float(spearmanr(X_explicar[col], shap_vals[:, i]).statistic))

    df_shap = pd.DataFrame({
        "feature": feature_names,
        "familia": [classificar_familia(c) for c in feature_names],
        "mean_abs_shap": mean_abs,
        "pct_impacto": (mean_abs / total_abs) * 100.0,
        "spearman_direcao": corrs,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    df_shap.insert(0, "rank", df_shap.index + 1)

    df_shap.to_csv(TAB_DIR / "shap_todas_features.csv", index=False)
    df_shap.head(30).to_csv(TAB_DIR / "shap_top30_features.csv", index=False)

    # 5. Auditoria agregada por família de descritores
    df_fam = (
        df_shap.groupby("familia", as_index=False)
        .agg(
            n_features=("feature", "count"),
            soma_mean_abs_shap=("mean_abs_shap", "sum"),
            pct_impacto_total=("pct_impacto", "sum"),
            impacto_medio_por_feature=("mean_abs_shap", "mean"),
        )
        .sort_values("pct_impacto_total", ascending=False)
        .reset_index(drop=True)
    )
    df_fam.to_csv(TAB_DIR / "shap_resumo_familias.csv", index=False)

    print("\nResumo de Impacto por Família de Descritores:")
    print(df_fam.round(4).to_string(index=False))

    print("\nTop 15 Features (direção > 0 favorece ABP):")
    print(df_shap.head(15)[["rank", "feature", "familia", "pct_impacto", "spearman_direcao"]].round(3).to_string(index=False))

    # 6. Gráficos
    plot_familias(df_fam, FIG_DIR / "01_shap_auditoria_familias.png")

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_vals, X_explicar, feature_names=feature_names, max_display=20, show=False)
    plt.title("SHAP Beeswarm (Top 20 Features — Valores Reais)", fontsize=13, pad=14)
    plt.gcf().savefig(FIG_DIR / "02_shap_beeswarm_top20.png", dpi=300, bbox_inches="tight")
    plt.close("all")

    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_vals, X_explicar, feature_names=feature_names, plot_type="bar", max_display=20, show=False
    )
    plt.title("SHAP Importância Global (Top 20 Features)", fontsize=13, pad=14)
    plt.gcf().savefig(FIG_DIR / "03_shap_bar_top20.png", dpi=300, bbox_inches="tight")
    plt.close("all")

    print(f"\n=== Sucesso! Figuras em {FIG_DIR.relative_to(ROOT)} e tabelas em {TAB_DIR.relative_to(ROOT)} ===")


if __name__ == "__main__":
    executar_analise_shap()
"""
src/features/tratamentoFeatures.py

Pipeline de divisão por homologia (40%) e seleção rigorosa de atributos (Anti-Data Leakage):
1. Divide Treino/Teste (80/20) via StratifiedGroupKFold usando 'cluster_id' (0% vazamento >= 40%).
2. Gera 'cv_fold' (5 folds agrupados por cluster) em meta_train.csv para alinhar LightGBM e ESM-2.
3. Executa o funil (Variância Zero -> Correlação |r| >= 0.85 -> RFECV com StratifiedGroupKFold).
4. Exporta:
   - X_train_limpo.csv / X_test_limpo.csv (para modelo final e teste cego)
   - X_train_bruto_split.csv / X_test_bruto_split.csv (para seleção aninhada no OOF sem viés otimista)
   - y_train.csv / y_test.csv
   - meta_train.csv / meta_test.csv (com id, sequence, label, cluster_id, group e cv_fold)
   - selected_features.json (lista final de colunas para projeção em benchmarks externos)
"""

import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedGroupKFold

warnings.filterwarnings("ignore", message=".*sklearn.utils.parallel.delayed.*")

# ==========================================
# Constantes e Diretórios
# ==========================================
ROOT = Path(__file__).resolve().parents[2]
PROC_DIR = ROOT / "data" / "processed"

LIMITE_CORR = 0.85
N_SPLITS_TEST = 5       # 1/5 = ~20% para teste cego agrupado por homologia
N_SPLITS_CV = 5         # 5-fold CV interno (compartilhado entre LightGBM e ESM-2)
MIN_FEATURES_RFECV = 25
RFECV_STEP = 1
SEED = 42

COLUNAS_METADADOS = {
    "id", "sequence", "label", "cluster_id", "cluster_40",
    "cluster", "group", "cv_fold", "index", "Unnamed: 0",
}


def separar_metadados(df: pd.DataFrame, alvo: str) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Separa explicitamente X (somente features numéricas), y (alvo) e metadados.
    Garante que colunas numéricas de controle (como cluster_id) nunca entrem em X.
    """
    cols_nao_numericas = set(df.select_dtypes(exclude=[np.number]).columns)
    cols_meta = [c for c in df.columns if c in COLUNAS_METADADOS or c in cols_nao_numericas]

    if alvo not in df.columns:
        raise ValueError(f"Coluna alvo '{alvo}' não encontrada no dataset.")

    y = df[alvo].copy()
    meta = df[cols_meta].copy()
    X = df.drop(columns=list(set(cols_meta) | {alvo})).copy()

    return X, y, meta


def _auditar_familias_features(colunas: list[str], titulo: str):
    """Mostra quantas features de cada família sobreviveram (auditoria de ruído OHE)."""
    contagem = {
        "Fisico-quimica": 0,
        "AAC": 0,
        "Z-scales": 0,
        "PC6": 0,
        "SOC_Eisenberg": 0,
        "OHE_N": 0,
        "OHE_C": 0,
    }
    for c in colunas:
        if c.startswith("AAC_"):
            contagem["AAC"] += 1
        elif c.startswith("Zscale_"):
            contagem["Z-scales"] += 1
        elif c.startswith("PC6_"):
            contagem["PC6"] += 1
        elif c.startswith("SOC_Eisenberg_") or c.startswith("PseAAC_"):
            contagem["SOC_Eisenberg"] += 1
        elif c.startswith("OHE_N"):
            contagem["OHE_N"] += 1
        elif c.startswith("OHE_C"):
            contagem["OHE_C"] += 1
        else:
            contagem["Fisico-quimica"] += 1

    resumo = " | ".join(f"{k}: {v}" for k, v in contagem.items())
    print(f"  -> [{titulo}] Total={len(colunas)} -> {resumo}")
    return contagem


def selecionar_atributos_funil(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups_train: pd.Series,
    X_eval: pd.DataFrame,
    corr_threshold: float = LIMITE_CORR,
    n_splits_cv: int = N_SPLITS_CV,
    min_features: int = MIN_FEATURES_RFECV,
    step: int = RFECV_STEP,
    seed: int = SEED,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Funil estritamente ajustado em X_train (com CV agrupado por homologia)
    e projetado em X_eval. Pode ser importado e chamado dentro de cada fold
    no treino do LightGBM para OOF 100% livre de viés otimista.
    """
    if verbose:
        print("\n--- Pipeline de Filtragem em Funil (Homologia 40%) ---")

    X_train = X_train.copy()
    X_eval = X_eval.copy()
    X_train.columns = X_train.columns.astype(str)
    X_eval.columns = X_eval.columns.astype(str)

    # ETAPA 1: Variância Zero
    variances = X_train.var()
    non_constant_cols = variances[variances > 0].index
    X_train_var = X_train[non_constant_cols]
    X_eval_var = X_eval[non_constant_cols]
    if verbose:
        print(
            f"  -> [Etapa 1 - Variância Zero] {len(variances) - len(non_constant_cols)} "
            f"colunas constantes removidas. Restam: {X_train_var.shape[1]}"
        )

    # ETAPA 2: Correlação (|r| >= corr_threshold)
    corr_matrix = X_train_var.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] >= corr_threshold)]

    X_train_uncorr = X_train_var.drop(columns=to_drop)
    X_eval_uncorr = X_eval_var.drop(columns=to_drop)
    if verbose:
        print(
            f"  -> [Etapa 2 - Correlação] {len(to_drop)} colunas removidas "
            f"(|r| >= {corr_threshold}). Restam: {X_train_uncorr.shape[1]}"
        )

    # ETAPA 3: RFECV com StratifiedGroupKFold (pré-computado para compatibilidade total)
    if verbose:
        print(
            f"  -> [Etapa 3 - RFECV] ExtraTrees + StratifiedGroupKFold({n_splits_cv}) "
            f"otimizando MCC..."
        )

    sgkf = StratifiedGroupKFold(n_splits=n_splits_cv, shuffle=True, random_state=seed)
    cv_splits = list(sgkf.split(X_train_uncorr, y_train, groups=groups_train))

    estimator = ExtraTreesClassifier(n_estimators=100, random_state=seed, n_jobs=None)
    rfecv = RFECV(
        estimator=estimator,
        step=step,
        min_features_to_select=min(min_features, X_train_uncorr.shape[1]),
        cv=cv_splits,
        scoring="matthews_corrcoef",
        n_jobs=-1,
    )
    rfecv.fit(X_train_uncorr, y_train)

    cols_finais = X_train_uncorr.columns[rfecv.support_].tolist()
    melhor_mcc = float(np.max(rfecv.cv_results_["mean_test_score"]))

    if verbose:
        print(
            f"  -> [Etapa 3 - RFECV] Ótimo: {len(cols_finais)} atributos | "
            f"Melhor MCC (CV 40%): {melhor_mcc:.4f}"
        )
        familias = _auditar_familias_features(cols_finais, "Sobreviventes RFECV")
    else:
        familias = {}

    info = {
        "n_features_inicial": int(X_train.shape[1]),
        "n_features_pos_var": int(X_train_var.shape[1]),
        "n_features_pos_corr": int(X_train_uncorr.shape[1]),
        "n_features_final": len(cols_finais),
        "melhor_mcc_rfecv_cv40": round(melhor_mcc, 5),
        "familias_sobreviventes": familias,
        "selected_features": cols_finais,
    }

    return X_train_uncorr[cols_finais], X_eval_uncorr[cols_finais], info


def dividir_por_homologia(
    X: pd.DataFrame, y: pd.Series, meta: pd.DataFrame, n_splits: int = N_SPLITS_TEST, seed: int = SEED
):
    """Divide ~80/20 garantindo 0% de compartilhamento de cluster_id entre treino e teste."""
    if "cluster_id" not in meta.columns:
        raise KeyError("Coluna 'cluster_id' ausente. Rode build_features.py atualizado primeiro.")

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_idx, test_idx = next(sgkf.split(X, y, groups=meta["cluster_id"]))

    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)
    meta_train = meta.iloc[train_idx].reset_index(drop=True)
    meta_test = meta.iloc[test_idx].reset_index(drop=True)

    # Auditoria rígida de vazamento
    clusters_tr = set(meta_train["cluster_id"])
    clusters_te = set(meta_test["cluster_id"])
    intersecao = clusters_tr & clusters_te
    assert len(intersecao) == 0, f"VAZAMENTO DETECTADO: {len(intersecao)} clusters em comum!"

    # Gera folds determinísticos de CV (40%) no treino para alinhar LightGBM e ESM-2
    sgkf_cv = StratifiedGroupKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=seed)
    meta_train["cv_fold"] = -1
    for fold, (_, val_idx) in enumerate(sgkf_cv.split(X_train, y_train, groups=meta_train["cluster_id"])):
        meta_train.loc[val_idx, "cv_fold"] = fold
    assert (meta_train["cv_fold"] >= 0).all(), "Erro ao atribuir cv_fold."

    return X_train, X_test, y_train, y_test, meta_train, meta_test


def processar_pipeline(entrada_csv: Path, alvo: str = "label"):
    if not entrada_csv.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {entrada_csv}")

    print("=== [1/3] Carregando e Separando Metadados ===")
    df = pd.read_csv(entrada_csv)
    print(f"  -> Shape original: {df.shape}")

    X, y, meta = separar_metadados(df, alvo)
    X = X.fillna(0)
    print(f"  -> Features numéricas: {X.shape[1]} | Metadados preservados: {list(meta.columns)}")

    print("\n=== [2/3] Split por Homologia (40% - StratifiedGroupKFold) ===")
    X_train, X_test, y_train, y_test, meta_train, meta_test = dividir_por_homologia(X, y, meta)

    print(
        f"  -> Treino: {len(X_train)} amostras "
        f"(Pos={int((y_train == 1).sum())}, Neg={int((y_train == 0).sum())}, "
        f"Clusters={meta_train['cluster_id'].nunique()})"
    )
    print(
        f"  -> Teste:  {len(X_test)} amostras "
        f"(Pos={int((y_test == 1).sum())}, Neg={int((y_test == 0).sum())}, "
        f"Clusters={meta_test['cluster_id'].nunique()})"
    )
    print("  -> Auditoria de vazamento: 0 clusters compartilhados entre Treino e Teste.")

    X_train_limpo, X_test_limpo, info = selecionar_atributos_funil(
        X_train=X_train,
        y_train=y_train,
        groups_train=meta_train["cluster_id"],
        X_eval=X_test,
    )

    print("\n=== [3/3] Exportando Matrizes, Metadados e Contrato de Features ===")
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    # Matrizes selecionadas (treino final + teste cego)
    X_train_limpo.to_csv(PROC_DIR / "X_train_limpo.csv", index=False)
    X_test_limpo.to_csv(PROC_DIR / "X_test_limpo.csv", index=False)

    # Matrizes brutas pós-split (para seleção aninhada por fold no OOF do LightGBM)
    X_train.to_csv(PROC_DIR / "X_train_bruto_split.csv", index=False)
    X_test.to_csv(PROC_DIR / "X_test_bruto_split.csv", index=False)

    # Alvos e Metadados alinhados 1-para-1 (Base A LightGBM + Base B ESM-2)
    y_train.to_csv(PROC_DIR / "y_train.csv", index=False)
    y_test.to_csv(PROC_DIR / "y_test.csv", index=False)
    meta_train.to_csv(PROC_DIR / "meta_train.csv", index=False)
    meta_test.to_csv(PROC_DIR / "meta_test.csv", index=False)

    with open(PROC_DIR / "selected_features.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    print(f"  -> X_train_limpo.csv: {X_train_limpo.shape} | X_test_limpo.csv: {X_test_limpo.shape}")
    print(f"  -> meta_train.csv (com cv_fold) e meta_test.csv salvos.")
    print(f"  -> selected_features.json salvo com {info['n_features_final']} atributos.")


if __name__ == "__main__":
    processar_pipeline(PROC_DIR / "dataset_treino.csv", alvo="label")
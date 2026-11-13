"""
src/features/tratamentoFeatures.py

Módulo agnóstico para divisão do dataset e filtragem rigorosa de atributos (Anti-Data Leakage).
Implementa um pipeline de SELEÇÃO DE ATRIBUTOS EM FUNIL otimizado para capturar sinergia:
Etapa 1: Filtro de Variância Zero (Remove colunas constantes)
Etapa 2: Filtro de Correlação (Remove redundância residual |r| >= 0.85)
Etapa 3: RFECV Multivariado (Otimiza MCC com ExtraTreesClassifier passo-a-passo)
"""

import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.ensemble import ExtraTreesClassifier

# Ignora os avisos de paralelismo interno do scikit-learn
warnings.filterwarnings("ignore", message=".*sklearn.utils.parallel.delayed.*")

# ==========================================
# Constantes e Diretórios
# ==========================================
ROOT = Path(__file__).resolve().parents[2]
PROC_DIR = ROOT / "data" / "processed"

LIMITE_CORR = 0.85
TEST_SIZE = 0.2
SEED = 42

def separar_metadados(df: pd.DataFrame, alvo: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Isola dinamicamente o alvo, dados numéricos e colunas de texto/metadados."""
    cols_nao_numericas = df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    for meta_comum in ["id", "index", "Unnamed: 0"]:
        if meta_comum in df.columns and meta_comum not in cols_nao_numericas:
            cols_nao_numericas.append(meta_comum)
            
    meta = df[cols_nao_numericas].copy() if cols_nao_numericas else pd.DataFrame(index=df.index)
    features = df.drop(columns=cols_nao_numericas)
    
    return features, meta

def selecionar_atributos_funil(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, 
                               corr_threshold: float = LIMITE_CORR):
    """
    Executa o funil de seleção sem mRMR estritamente no conjunto de treino.
    Projeta as colunas sobreviventes no conjunto de teste.
    """
    print("\n--- Pipeline de Filtragem em Funil (Espaço de Treino) ---")
    
    # Garantia de tipagem
    X_train.columns = X_train.columns.astype(str)
    X_test.columns = X_test.columns.astype(str)

    # -------------------------------------------------------------
    # ETAPA 1: Remover Colunas Constantes (Variância Zero)
    # -------------------------------------------------------------
    variances = X_train.var()
    non_constant_cols = variances[variances > 0].index
    X_train_var = X_train[non_constant_cols]
    X_test_var = X_test[non_constant_cols]
    removidas_constantes = len(variances) - len(non_constant_cols)
    print(f"  -> [Etapa 1 - Variância Zero] {removidas_constantes} colunas constantes removidas.")

    # -------------------------------------------------------------
    # ETAPA 2: Remoção de Multicolinearidade Residual (|r| >= corr_threshold)
    # -------------------------------------------------------------
    corr_matrix = X_train_var.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] >= corr_threshold)]

    X_train_uncorr = X_train_var.drop(columns=to_drop)
    X_test_uncorr = X_test_var.drop(columns=to_drop)
    print(f"  -> [Etapa 2 - Correlação] {len(to_drop)} colunas correlacionadas (|r| >= {corr_threshold}) removidas. Restam: {X_train_uncorr.shape[1]} colunas.")

    # -------------------------------------------------------------
    # ETAPA 3: Seleção Multivariada com RFECV (Otimizando MCC)
    # -------------------------------------------------------------
    print("  -> [Etapa 3 - RFECV] Avaliando blocos com ExtraTreesClassifier para otimizar MCC...")
    
    # MUDANÇA AQUI: n_jobs=None no ExtraTrees para evitar conflito de paralelismo
    estimator = ExtraTreesClassifier(n_estimators=100, random_state=SEED, n_jobs=None)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    # O n_jobs=-1 fica APENAS no RFECV
    rfecv = RFECV(
        estimator=estimator,
        step=1,                      
        min_features_to_select=30,   
        cv=cv,
        scoring='matthews_corrcoef', 
        n_jobs=-1
    )

    rfecv.fit(X_train_uncorr, y_train)

    cols_finais = X_train_uncorr.columns[rfecv.support_]
    print(f"  -> [Etapa 3 - RFECV] Sucesso! Número ótimo de atributos encontrados: {len(cols_finais)}")

    # -------------------------------------------------------------
    # Matrizes Finais
    # -------------------------------------------------------------
    X_train_final = X_train_uncorr[cols_finais]
    X_test_final = X_test_uncorr[cols_finais]

    return X_train_final, X_test_final


def processar_pipeline(entrada_csv: Path, alvo: str = "label"):
    if not entrada_csv.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {entrada_csv}")

    print("=== Iniciando Engenharia e Seleção de Atributos (Sem mRMR) ===")
    df = pd.read_csv(entrada_csv)
    print(f"  -> Shape original: {df.shape}")

    if alvo not in df.columns:
        raise ValueError(f"Coluna alvo '{alvo}' não encontrada no dataset.")

    df_features, df_meta = separar_metadados(df, alvo)
    X = df_features.drop(columns=[alvo])
    y = df_features[alvo]

    X = X.fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )
    print(f"  -> Treino: {X_train.shape[0]} amostras | Teste Cego: {X_test.shape[0]} amostras.")

    X_train_limpo, X_test_limpo = selecionar_atributos_funil(X_train, y_train, X_test)

    PROC_DIR.mkdir(parents=True, exist_ok=True)
    X_train_limpo.to_csv(PROC_DIR / "X_train_limpo.csv", index=False)
    X_test_limpo.to_csv(PROC_DIR / "X_test_limpo.csv", index=False)
    y_train.to_csv(PROC_DIR / "y_train.csv", index=False)
    y_test.to_csv(PROC_DIR / "y_test.csv", index=False)

    print("\n=== Pipeline Finalizado com Sucesso ===")
    print(f"  -> Shape Final - Treino: {X_train_limpo.shape}")
    print(f"  -> Shape Final - Teste:  {X_test_limpo.shape}")

if __name__ == "__main__":
    processar_pipeline(PROC_DIR / "dataset_treino.csv", alvo="label")
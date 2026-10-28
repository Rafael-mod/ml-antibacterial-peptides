"""
src/features/tratamentoColunas.py

Módulo agnóstico para divisão do dataset e filtragem rigorosa de atributos (Anti-Data Leakage).
Implementa um pipeline híbrido de SELEÇÃO DE ATRIBUTOS EM FUNIL:
Etapa 1: ANOVA (Top 500 features com maior sinal)
Etapa 2: Filtro de Correlação (Remove redundância |r| >= 0.85)
Etapa 3: RFECV Multivariado (Otimiza MCC com ExtraTreesClassifier)
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_classif, RFECV
from sklearn.ensemble import ExtraTreesClassifier

# ==========================================
# Constantes e Diretórios
# ==========================================
ROOT = Path(__file__).resolve().parents[2]
PROC_DIR = ROOT / "data" / "processed"

TOP_K_ANOVA = 500
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
                               top_k_anova: int = TOP_K_ANOVA, corr_threshold: float = LIMITE_CORR):
    """
    Executa o funil de seleção em 3 etapas estritamente no conjunto de treino.
    Projeta as colunas sobreviventes no conjunto de teste.
    """
    print("\n--- Pipeline de Filtragem em Funil (Espaço de Treino) ---")
    
    # -------------------------------------------------------------
    # ETAPA 0: Remover Colunas Constantes (Variância Zero)
    # -------------------------------------------------------------
    variances = X_train.var()
    non_constant_cols = variances[variances > 0].index
    X_train = X_train[non_constant_cols]
    X_test = X_test[non_constant_cols]
    removidas_constantes = len(variances) - len(non_constant_cols)
    print(f"  -> [Etapa 0 - Variância Zero] {removidas_constantes} colunas constantes removidas.")

    # -------------------------------------------------------------
    # ETAPA 1: Funil Univariado via ANOVA (SelectKBest f_classif)
    # -------------------------------------------------------------
    k_actual = min(top_k_anova, X_train.shape[1])
    anova = SelectKBest(score_func=f_classif, k=k_actual)
    anova.fit(X_train, y_train)  # Fit exclusivo no treino

    cols_anova = X_train.columns[anova.get_support()]
    X_train_anova = X_train[cols_anova]
    X_test_anova = X_test[cols_anova]
    print(f"  -> [Etapa 1 - ANOVA] Reduzido de {X_train.shape[1]} para as Top {len(cols_anova)} features com maior sinal.")

    # -------------------------------------------------------------
    # ETAPA 2: Remoção de Multicolinearidade (|r| >= corr_threshold)
    # -------------------------------------------------------------
    corr_matrix = X_train_anova.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] >= corr_threshold)]

    X_train_uncorr = X_train_anova.drop(columns=to_drop)
    X_test_uncorr = X_test_anova.drop(columns=to_drop)
    print(f"  -> [Etapa 2 - Correlação] {len(to_drop)} colunas redundantes (|r| >= {corr_threshold}) removidas. Restam: {X_train_uncorr.shape[1]} colunas.")

    # -------------------------------------------------------------
    # ETAPA 3: Seleção Multivariada com RFECV (Otimizando MCC)
    # -------------------------------------------------------------
    print("  -> [Etapa 3 - RFECV] Avaliando blocos com ExtraTreesClassifier para otimizar MCC...")
    estimator = ExtraTreesClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    rfecv = RFECV(
        estimator=estimator,
        step=0.05,                   # Elimina 5% das piores colunas por iteração
        min_features_to_select=50,   # Garante um limite mínimo de features
        cv=cv,
        scoring='matthews_corrcoef', # Foco direto na métrica principal (MCC)
        n_jobs=-1
    )

    rfecv.fit(X_train_uncorr, y_train)  # Fit exclusivo no treino

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

    print("=== Iniciando Engenharia e Seleção de Atributos ===")
    df = pd.read_csv(entrada_csv)
    print(f"  -> Shape original: {df.shape}")

    if alvo not in df.columns:
        raise ValueError(f"Coluna alvo '{alvo}' não encontrada no dataset.")

    # 1. Isolamento Automático (Remove IDs e Textos)
    df_features, df_meta = separar_metadados(df, alvo)
    X = df_features.drop(columns=[alvo])
    y = df_features[alvo]

    # Preenche NaNs eventuais com 0 antes de separar
    X = X.fillna(0)

    # 2. Split Holdout Estratificado (Treino / Teste)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )
    print(f"  -> Treino: {X_train.shape[0]} amostras | Teste Cego: {X_test.shape[0]} amostras.")

    # 3. Pipeline de Seleção (Estratégia de Funil Híbrido)
    X_train_limpo, X_test_limpo = selecionar_atributos_funil(X_train, y_train, X_test)

    # 4. Salvamento Vetorial das Matrizes Otimizadas
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
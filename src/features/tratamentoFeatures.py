"""
src/features/tratamentoColunas.py

Módulo agnóstico para divisão do dataset e filtragem rigorosa de atributos (Anti-Data Leakage).
Suporta crescimento ilimitado do espaço de features (AAC, PseAAC, OHE, etc.).
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).resolve().parents[2]
PROC_DIR = ROOT / "data" / "processed"

LIMITE_CORR = 0.90
LIMITE_P_VALOR = 0.05
TEST_SIZE = 0.2
SEED = 42

def separar_metadados(df: pd.DataFrame, alvo: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Isola dinamicamente o alvo, dados numéricos e colunas de texto/metadados."""
    # Identifica colunas que não são numéricas (ex: 'sequence', 'id' textual)
    cols_nao_numericas = df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    # Adiciona colunas conhecidas de metadados se existirem, mesmo se forem numéricas
    for meta_comum in ["id", "index", "Unnamed: 0"]:
        if meta_comum in df.columns and meta_comum not in cols_nao_numericas:
            cols_nao_numericas.append(meta_comum)
            
    meta = df[cols_nao_numericas].copy() if cols_nao_numericas else pd.DataFrame(index=df.index)
    features = df.drop(columns=cols_nao_numericas)
    
    return features, meta

def remover_variancia_zero(X_train: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Remove colunas constantes independentemente do tamanho do dataframe."""
    sel = VarianceThreshold(threshold=0.0)
    sel.fit(X_train)
    
    colunas_mantidas = X_train.columns[sel.get_support()].tolist()
    removidas = len(X_train.columns) - len(colunas_mantidas)
    
    print(f"  -> {removidas} colunas constantes (variância zero) removidas.")
    return X_train[colunas_mantidas], colunas_mantidas

def remover_alta_correlacao(X_train: pd.DataFrame, limite: float = LIMITE_CORR) -> tuple[pd.DataFrame, list[str]]:
    """Remove features altamente correlacionadas com conversão de precisão para otimizar RAM."""
    # Reduz precisão para float32 temporariamente para acelerar Spearman em datasets largos
    X_temp = X_train.astype(np.float32)
    
    corr_matrix = X_temp.corr(method="spearman").abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    colunas_remover = [col for col in upper.columns if any(upper[col] >= limite)]
    colunas_mantidas = [col for col in X_train.columns if col not in colunas_remover]
    
    print(f"  -> {len(colunas_remover)} colunas removidas por alta redundância (|r| >= {limite}).")
    return X_train[colunas_mantidas], colunas_mantidas

def aplicar_mann_whitney(X_train: pd.DataFrame, y_train: pd.Series, limite_p: float = LIMITE_P_VALOR) -> tuple[pd.DataFrame, list[str]]:
    """Teste estatístico de sobrevivência tolerante a distribuições assimétricas extremas."""
    pos = X_train[y_train == 1]
    neg = X_train[y_train == 0]
    
    colunas_mantidas = []
    
    for col in X_train.columns:
        # Prevenção de quebra do loop por colunas com variância local zero ou NaNs
        if pos[col].nunique() <= 1 and neg[col].nunique() <= 1:
            continue
            
        try:
            _, p_valor = mannwhitneyu(pos[col].dropna(), neg[col].dropna(), alternative="two-sided")
            if p_valor < limite_p:
                colunas_mantidas.append(col)
        except Exception:
            # Falhas numéricas silenciosas descartam a coluna (feature inviável)
            continue
            
    removidas = len(X_train.columns) - len(colunas_mantidas)
    print(f"  -> {removidas} colunas removidas (p >= {limite_p} ou erro numérico).")
    print(f"  -> {len(colunas_mantidas)} features biológicas estatisticamente relevantes mantidas.")
    
    return X_train[colunas_mantidas], colunas_mantidas

def processar_pipeline(entrada_csv: Path, alvo: str = "label"):
    if not entrada_csv.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {entrada_csv}")

    df = pd.read_csv(entrada_csv)
    print(f"  -> Shape original: {df.shape}")

    if alvo not in df.columns:
        raise ValueError(f"Coluna alvo '{alvo}' não encontrada.")

    # 1. Isolamento Automático
    df_features, df_meta = separar_metadados(df, alvo)
    X = df_features.drop(columns=[alvo])
    y = df_features[alvo]

    # Preenche NaNs eventuais gerados por extratores com 0 antes do pipeline
    X = X.fillna(0)

    # 2. Split Holdout Estratificado
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )
    print(f"  -> Treino: {X_train.shape[0]} amostras | Teste Cego: {X_test.shape[0]} amostras.")

    # 3. Pipeline Anti-Leakage (Somente Treino)
    print("\n--- Pipeline de Filtragem (Espaço de Treino) ---")
    X_train_limpo, cols_validas = remover_variancia_zero(X_train)
    X_train_limpo, cols_validas = remover_alta_correlacao(X_train_limpo, limite=LIMITE_CORR)
    X_train_limpo, colunas_finais = aplicar_mann_whitney(X_train_limpo, y_train, limite_p=LIMITE_P_VALOR)

    # 4. Projeção das features validadas no Teste
    X_test_limpo = X_test[colunas_finais]

    # 5. Salvamento Vetorial
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    X_train_limpo.to_csv(PROC_DIR / "X_train_limpo.csv", index=False)
    X_test_limpo.to_csv(PROC_DIR / "X_test_limpo.csv", index=False)
    y_train.to_csv(PROC_DIR / "y_train.csv", index=False)
    y_test.to_csv(PROC_DIR / "y_test.csv", index=False)

    print("\n=== Pipeline Finalizado ===")
    print(f"  -> Shape Final - Treino: {X_train_limpo.shape}")
    print(f"  -> Shape Final - Teste:  {X_test_limpo.shape}")

if __name__ == "__main__":
    print("=== Iniciando Engenharia e Seleção de Atributos ===")
    processar_pipeline(PROC_DIR / "dataset_treino.csv", alvo="label")
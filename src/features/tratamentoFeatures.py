"""
src/features/tratamentoColunas.py

Módulo para divisão do dataset e filtragem rigorosa de atributos (Anti-Data Leakage):
1. Divisão Holdout (80% Treino / 20% Teste Cego).
2. Filtro Não-Supervisionado (Treino): Remoção de features com variância nula.
3. Filtro Não-Supervisionado (Treino): Remoção de features altamente correlacionadas (Spearman > 0.90).
4. Filtro Supervisionado (Treino): Teste de Mann-Whitney U (p < 0.05).
5. Aplicação das colunas sobreviventes no Teste Cego.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import mannwhitneyu

# Configuração de caminhos e constantes
ROOT = Path(__file__).resolve().parents[2]
PROC_DIR = ROOT / "data" / "processed"

LIMITE_CORR = 0.90
LIMITE_P_VALOR = 0.05
TEST_SIZE = 0.2
SEED = 42

def separar_metadados(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separa as colunas de metadados das features numéricas."""
    colunas_meta = [c for c in ["id", "sequence", "classe"] if c in df.columns]
    meta = df[colunas_meta] if colunas_meta else pd.DataFrame(index=df.index)
    features = df.drop(columns=colunas_meta)
    return features, meta

def remover_variancia_zero(X_train: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Remove colunas que possuem o mesmo valor em todas as amostras (constantes)."""
    # Identifica colunas numéricas
    num_cols = X_train.select_dtypes(include=[np.number]).columns
    
    sel = VarianceThreshold(threshold=0.0)
    sel.fit(X_train[num_cols])
    
    colunas_mantidas = [c for c, ok in zip(num_cols, sel.get_support()) if ok]
    colunas_removidas = [c for c in num_cols if c not in colunas_mantidas]
    
    print(f"  -> {len(colunas_removidas)} colunas constantes/variância zero removidas.")
    return X_train[colunas_mantidas], colunas_mantidas

def remover_alta_correlacao(X_train: pd.DataFrame, limite: float = LIMITE_CORR) -> tuple[pd.DataFrame, list[str]]:
    """Calcula Spearman no treino e remove features redundantes."""
    corr_matrix = X_train.corr(method="spearman").abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    colunas_remover = [col for col in upper.columns if any(upper[col] >= limite)]
    colunas_mantidas = [col for col in X_train.columns if col not in colunas_remover]
    
    if colunas_remover:
        print(f"  -> {len(colunas_remover)} colunas removidas por alta correlação (|r| >= {limite}).")
    else:
        print(f"  -> Nenhuma coluna removida por correlação.")
        
    return X_train[colunas_mantidas], colunas_mantidas

def aplicar_mann_whitney(X_train: pd.DataFrame, y_train: pd.Series, limite_p: float = LIMITE_P_VALOR) -> tuple[pd.DataFrame, list[str]]:
    """Aplica o teste de Mann-Whitney U para reter apenas features que separam as classes."""
    pos = X_train[y_train == 1]
    neg = X_train[y_train == 0]
    
    colunas_mantidas = []
    colunas_descartadas = []
    
    for col in X_train.columns:
        # Pula colunas que não têm variância suficiente dentro de um dos grupos para o teste
        if pos[col].nunique() <= 1 and neg[col].nunique() <= 1:
            colunas_descartadas.append(col)
            continue
            
        _, p_valor = mannwhitneyu(pos[col], neg[col], alternative="two-sided")
        
        if p_valor < limite_p:
            colunas_mantidas.append(col)
        else:
            colunas_descartadas.append(col)
            
    print(f"  -> {len(colunas_descartadas)} colunas removidas pelo Mann-Whitney U (p >= {limite_p}).")
    print(f"  -> {len(colunas_mantidas)} colunas selecionadas como estatisticamente significativas.")
    
    return X_train[colunas_mantidas], colunas_mantidas

def main(entrada: Path = PROC_DIR / "dataset_bruto.csv"):
    print("=== Iniciando Divisão e Filtragem Anti-Vazamento ===")
    
    if not entrada.exists():
        raise FileNotFoundError(f"Ficheiro não encontrado: {entrada}")

    df = pd.read_csv(entrada)
    print(f"  -> Shape original do dataset: {df.shape}")

    if "label" not in df.columns:
        raise ValueError("A coluna alvo 'label' não foi encontrada no dataset.")

    # 1. Separação de X e y, isolando metadados
    df_features, df_meta = separar_metadados(df)
    X = df_features.drop(columns=["label"])
    y = df_features["label"]

    # 2. Split Holdout Estratificado
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )
    print(f"  -> Split concluído: Treino ({X_train.shape[0]} amostras), Teste ({X_test.shape[0]} amostras).")

    # 3. Pipeline de Filtros aplicados APENAS no Treino
    print("\n--- Aplicando Filtros no Conjunto de Treino ---")
    X_train_limpo, cols_validas = remover_variancia_zero(X_train)
    X_train_limpo, cols_validas = remover_alta_correlacao(X_train_limpo, limite=LIMITE_CORR)
    X_train_limpo, colunas_finais = aplicar_mann_whitney(X_train_limpo, y_train, limite_p=LIMITE_P_VALOR)

    # 4. Aplicação das colunas sobreviventes no Teste Cego
    X_test_limpo = X_test[colunas_finais]

    # 5. Exportação
    X_train_limpo.to_csv(PROC_DIR / "X_train_limpo.csv", index=False)
    X_test_limpo.to_csv(PROC_DIR / "X_test_limpo.csv", index=False)
    y_train.to_csv(PROC_DIR / "y_train.csv", index=False)
    y_test.to_csv(PROC_DIR / "y_test.csv", index=False)

    print("\n=== Sucesso! ===")
    print(f"  -> Shape final de Treino: {X_train_limpo.shape}")
    print(f"  -> Shape final de Teste:  {X_test_limpo.shape}")
    print(f"  -> Arquivos limpos guardados em: {PROC_DIR.relative_to(ROOT)}")

if __name__ == "__main__":
    main()
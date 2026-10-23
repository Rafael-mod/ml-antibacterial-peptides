"""
src/features/preprocessamento.py

Módulo para limpeza e pré-processamento de features:
1. Remoção dinâmica de features altamente correlacionadas (redundantes) via Spearman.
2. Remoção de features One-Hot com variância nula.
3. Criação de um pipeline (ColumnTransformer) para padronização e transformação de distribuições assimétricas.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import PowerTransformer, StandardScaler

# Configuração de caminhos e constantes
ROOT = Path(__file__).resolve().parents[2]
PROC_DIR = ROOT / "data" / "processed"

LIMITE_CORR = 0.90
LIMITE_SKEW = 1.0


def separar_grupos(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    """Separa as colunas do dataset nos três grupos de features."""
    aac = [c for c in df.columns if c.startswith("AAC_")]
    oh = [c for c in df.columns if c.startswith("OH_")]
    fis = [c for c in df.columns if c not in ("id", "sequence", "label", "classe") and c not in aac and c not in oh]
    
    return fis, aac, oh


def remover_alta_correlacao(df: pd.DataFrame, limite: float = LIMITE_CORR) -> pd.DataFrame:
    """
    Calcula a matriz de correlação de Spearman para features contínuas
    e remove uma feature de cada par com correlação absoluta >= limite.
    """
    fis, aac, _ = separar_grupos(df)
    cols_continuas = fis + aac
    
    # Calcula a matriz de correlação absoluta
    corr_matrix = df[cols_continuas].corr(method="spearman").abs()
    
    # Seleciona apenas o triângulo superior da matriz para não duplicar pares
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Identifica colunas a serem removidas
    colunas_remover = [col for col in upper.columns if any(upper[col] >= limite)]
    
    if colunas_remover:
        print(f"  -> {len(colunas_remover)} colunas removidas por alta correlação (|r| >= {limite}):")
        # Exibe as colunas de forma compacta (grupos de 5)
        for i in range(0, len(colunas_remover), 5):
            print(f"     {', '.join(colunas_remover[i:i+5])}")
        return df.drop(columns=colunas_remover)
    
    print(f"  -> Nenhuma coluna removida (Nenhuma correlação >= {limite}).")
    return df


def remover_one_hot_variancia_zero(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Identifica e remove features One-Hot que têm sempre o mesmo valor (variância nula)."""
    _, _, oh = separar_grupos(df)
    if not oh:
        return df, []

    sel = VarianceThreshold(threshold=0.0).fit(df[oh])
    removidas = [c for c, ok in zip(oh, sel.get_support()) if not ok]
    
    print(f"  -> Colunas One-Hot com variância zero removidas: {len(removidas)} de {len(oh)}")
    
    return df.drop(columns=removidas), removidas


def colunas_assimetricas(df: pd.DataFrame, cols: list[str], limite: float = LIMITE_SKEW) -> list[str]:
    """Identifica as features numéricas que ultrapassam o limite de assimetria (skewness)."""
    skew = df[cols].skew()
    alvo = skew[skew.abs() > limite].sort_values(key=abs, ascending=False)
    
    if len(alvo) > 0:
        print(f"  -> Colunas com |skewness| > {limite}:\n{alvo.round(2).to_string()}")
    else:
        print(f"  -> Nenhuma coluna excedeu o limite de assimetria de {limite}.")
        
    return list(alvo.index)


def criar_preprocessador(df: pd.DataFrame, limite_skew: float = LIMITE_SKEW) -> ColumnTransformer:
    """
    Cria um pipeline de transformação ajustado para Modelos Lineares, SVM e Redes Neuronais.
    Nota: Deve ser ajustado (.fit) EXCLUSIVAMENTE nos dados de treino para evitar vazamento de dados.
    """
    fis, aac, _ = separar_grupos(df)
    assimetricas = colunas_assimetricas(df, fis, limite_skew)
    normais = [c for c in fis if c not in assimetricas]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("yeo_johnson", PowerTransformer(method="yeo-johnson", standardize=True), assimetricas),
            ("padroniza", StandardScaler(), normais + aac),
        ],
        remainder="passthrough", # As colunas One-Hot (0/1) e Labels passam sem alteração
        verbose_feature_names_out=False,
    )
    
    return preprocessor.set_output(transform="pandas")


def main(entrada: Path = PROC_DIR / "dataset_treino.csv", 
         saida: Path = PROC_DIR / "dataset_treino_limpo.csv") -> pd.DataFrame:
    
    print("=== Iniciando Pré-processamento e Limpeza ===")
    
    if not entrada.exists():
        raise FileNotFoundError(f"Ficheiro não encontrado: {entrada}")

    df = pd.read_csv(entrada)
    print(f"  -> Shape original: {df.shape}")

    # Remove correlações dinamicamente
    df = remover_alta_correlacao(df, limite=LIMITE_CORR)
    
    # Remove colunas one-hot inativas
    df, _ = remover_one_hot_variancia_zero(df)

    df.to_csv(saida, index=False)
    print(f"\n=== Sucesso! ===")
    print(f"  -> Shape final: {df.shape}")
    print(f"  -> Dataset limpo guardado em: {saida.relative_to(ROOT)}")
    
    return df


if __name__ == "__main__":
    df_limpo = main()
"""
src/features/preprocessamento.py

Módulo para limpeza e pré-processamento de features:
1. Remoção de features altamente correlacionadas (redundantes).
2. Remoção de features One-Hot com variância nula.
3. Criação de um pipeline (ColumnTransformer) para padronização e transformação de distribuições assimétricas.
"""

from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import PowerTransformer, StandardScaler

# Configuração de caminhos e constantes
ROOT = Path(__file__).resolve().parents[2]
PROC_DIR = ROOT / "data" / "processed"

COLS_REMOVER = ["peso_molecular", "ponto_isoeletrico"]
LIMITE_SKEW = 1.0


def separar_grupos(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    """Separa as colunas do dataset nos três grupos de features."""
    aac = [c for c in df.columns if c.startswith("AAC_")]
    oh = [c for c in df.columns if c.startswith("OH_")]
    fis = [c for c in df.columns if c not in ("sequence", "label", "classe") and c not in aac and c not in oh]
    
    return fis, aac, oh


def remover_colunas_redundantes(df: pd.DataFrame, colunas: list[str] = COLS_REMOVER) -> pd.DataFrame:
    """Remove colunas previamente identificadas como redundantes por alta correlação."""
    existentes = [c for c in colunas if c in df.columns]
    ausentes = [c for c in colunas if c not in df.columns]
    
    if ausentes:
        print(f"  -> Aviso: Colunas a remover não encontradas (ignoradas): {ausentes}")
    
    if existentes:
        print(f"  -> Colunas removidas por redundância: {existentes}")
        return df.drop(columns=existentes)
    
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


def criar_preprocessador(df: pd.DataFrame, limite: float = LIMITE_SKEW) -> ColumnTransformer:
    """
    Cria um pipeline de transformação ajustado para Modelos Lineares, SVM e Redes Neuronais.
    Nota: Deve ser ajustado (.fit) EXCLUSIVAMENTE nos dados de treino para evitar vazamento de dados.
    """
    fis, aac, _ = separar_grupos(df)
    assimetricas = colunas_assimetricas(df, fis, limite)
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

    df = remover_colunas_redundantes(df)
    df, _ = remover_one_hot_variancia_zero(df)

    df.to_csv(saida, index=False)
    print(f"\n=== Sucesso! ===")
    print(f"  -> Shape final: {df.shape}")
    print(f"  -> Dataset limpo guardado em: {saida.relative_to(ROOT)}")
    
    return df


if __name__ == "__main__":
    df_limpo = main()
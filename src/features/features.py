"""
src/features/build_features.py

Engenharia de Features para Peptídeos Antimicrobianos:
1. Lê 'dataset_positivo.csv' e 'dataset_negativo.csv' de 'data/processed/'.
2. Unifica os dados tratando a coluna 'group' (atribui 'negativo' aos exemplos sem grupo).
3. Extrai propriedades físico-químicas (PM, Carga, pI, GRAVY, Momento Hidrofóbico, etc.).
4. Extrai a Composição de Aminoácidos (AAC - 20 features).
5. Gera One-Hot Encoding posicional com padding (50 x 20 = 1000 features).
6. Salva 'dataset_unificado.csv' e 'dataset_treino.csv' em 'data/processed/'.

Uso:
    python src/features/build_features.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis

# ==========================================
# Diretórios do Projeto
# ==========================================
ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"

CSV_POS = PROC / "dataset_positivo.csv"
CSV_NEG = PROC / "dataset_negativo.csv"

OUT_UNIFICADO = PROC / "dataset_unificado.csv"
OUT_TREINO = PROC / "dataset_treino.csv"

# ==========================================
# Constantes Biológicas e Químicas
# ==========================================
AMINOACIDOS = list("ACDEFGHIKLMNPQRSTVWY")

# Escala de Eisenberg (consenso normalizado) para o momento hidrofóbico
EISENBERG = {
    "A": 0.62, "R": -2.53, "N": -0.78, "D": -0.90, "C": 0.29, "Q": -0.85,
    "E": -0.74, "G": 0.48, "H": -0.40, "I": 1.38, "L": 1.06, "K": -1.50,
    "M": 0.64, "F": 1.19, "P": 0.12, "S": -0.18, "T": -0.05, "W": 0.81,
    "Y": 0.26, "V": 1.08,
}

HIDROFOBICOS = set("AILMFVW")
POLARES = set("STNQ")
POSITIVOS = set("KRH")
NEGATIVOS = set("DE")


# ==========================================
# Funções Auxiliares
# ==========================================
def _validar(seq: str) -> str:
    seq = str(seq).strip().upper()
    invalidos = set(seq) - set(AMINOACIDOS)
    if not seq or invalidos:
        raise ValueError(f"Sequência inválida ({seq!r}); caracteres não canônicos: {invalidos}")
    return seq


def _momento_hidrofobico(seq: str, angulo_graus: float = 100.0) -> float:
    """Momento hidrofóbico médio (100° = alfa-hélice)."""
    ang = np.deg2rad(angulo_graus) * np.arange(len(seq))
    h = np.array([EISENBERG[aa] for aa in seq])
    return float(np.hypot((h * np.cos(ang)).sum(), (h * np.sin(ang)).sum()) / len(seq))


# ==========================================
# Calculadores de Features
# ==========================================
def calcular_propriedades_fisicoquimicas(df: pd.DataFrame, col_seq: str = "sequence", ph: float = 7.4) -> pd.DataFrame:
    linhas = []
    for seq in df[col_seq]:
        seq = _validar(seq)
        n = len(seq)
        pa = ProteinAnalysis(seq)
        contagem = pa.count_amino_acids()
        helice, volta, folha = pa.secondary_structure_fraction()

        pct = lambda aa: 100.0 * contagem[aa] / n
        frac = lambda conjunto: sum(contagem[aa] for aa in conjunto) / n

        linhas.append({
            "comprimento": n,
            "peso_molecular": pa.molecular_weight(),
            "carga_liquida": pa.charge_at_pH(ph),
            "ponto_isoeletrico": pa.isoelectric_point(),
            "gravy": pa.gravy(),
            "aromaticidade": pa.aromaticity(),
            "indice_instabilidade": pa.instability_index(),
            "indice_alifatico": pct("A") + 2.9 * pct("V") + 3.9 * (pct("I") + pct("L")),
            "frac_helice": helice,
            "frac_volta": volta,
            "frac_folha": folha,
            "frac_hidrofobica": frac(HIDROFOBICOS),
            "frac_polar": frac(POLARES),
            "frac_positiva": frac(POSITIVOS),
            "frac_negativa": frac(NEGATIVOS),
            "momento_hidrofobico": _momento_hidrofobico(seq),
        })
    return pd.DataFrame(linhas, index=df.index)


def calcular_aac(df: pd.DataFrame, col_seq: str = "sequence") -> pd.DataFrame:
    linhas = []
    for seq in df[col_seq]:
        seq = _validar(seq)
        contagem = ProteinAnalysis(seq).count_amino_acids()
        linhas.append({f"AAC_{aa}": contagem[aa] / len(seq) for aa in AMINOACIDOS})
    return pd.DataFrame(linhas, index=df.index)


def one_hot_encoding(df: pd.DataFrame, col_seq: str = "sequence", max_len: int = 50) -> pd.DataFrame:
    """Colunas OH_p01_A ... OH_p50_Y; sequências curtas são completadas com zeros (padding)."""
    indice_aa = {aa: i for i, aa in enumerate(AMINOACIDOS)}
    seqs = [_validar(s) for s in df[col_seq]]
    if max(len(s) for s in seqs) > max_len:
        raise ValueError(f"Há sequências maiores que max_len={max_len}")

    matriz = np.zeros((len(seqs), max_len, len(AMINOACIDOS)), dtype=np.uint8)
    for i, seq in enumerate(seqs):
        for j, aa in enumerate(seq):
            matriz[i, j, indice_aa[aa]] = 1

    colunas = [f"OH_p{p + 1:02d}_{aa}" for p in range(max_len) for aa in AMINOACIDOS]
    return pd.DataFrame(matriz.reshape(len(seqs), -1), columns=colunas, index=df.index)


# ==========================================
# Montagem e Execução
# ==========================================
def construir_features(
    caminho_pos: Path = CSV_POS,
    caminho_neg: Path = CSV_NEG,
    seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    
    print("=== [1/3] Carregando Datasets Processados ===")
    if not caminho_pos.exists() or not caminho_neg.exists():
        raise FileNotFoundError(
            f"Arquivos CSV não encontrados em {PROC}. "
            "Execute 'python src/data/create_csvs.py' primeiro."
        )

    pos = pd.read_csv(caminho_pos)
    neg = pd.read_csv(caminho_neg)

    # Garante a coluna 'group' no unificado
    if "group" not in neg.columns:
        neg["group"] = "negativo"

    print(f"  -> Positivos lidos: {len(pos)} | Negativos lidos: {len(neg)}")

    # Unificação e remoção de duplicatas
    unificado = pd.concat([pos, neg], ignore_index=True)
    unificado = unificado.drop_duplicates(subset="sequence")
    unificado = unificado.sample(frac=1, random_state=seed).reset_index(drop=True)

    print("\n=== [2/3] Extraindo Features Biológicas e Numéricas ===")
    print("  -> Calculando propriedades físico-químicas...")
    df_prop = calcular_propriedades_fisicoquimicas(unificado)
    
    print("  -> Calculando composição de aminoácidos (AAC)...")
    df_aac = calcular_aac(unificado)
    
    print("  -> Gerando One-Hot Encoding posicional...")
    df_ohe = one_hot_encoding(unificado)

    # 'group' fica fora das features do dataset de treino para evitar vazamento
    treino = pd.concat(
        [
            unificado[["id", "sequence", "label"]],
            df_prop,
            df_aac,
            df_ohe,
        ],
        axis=1,
    )

    return unificado, treino


def main():
    unificado, treino = construir_features()

    PROC.mkdir(parents=True, exist_ok=True)
    unificado.to_csv(OUT_UNIFICADO, index=False)
    treino.to_csv(OUT_TREINO, index=False)

    print("\n=== [3/3] Arquivos Salvos em data/processed/ ===")
    print(f"  -> Unificado: {OUT_UNIFICADO.name} | Dimensões: {unificado.shape}")
    print(f"  -> Treino:    {OUT_TREINO.name} | Dimensões: {treino.shape}")
    print("\n✅ Engenharia de features concluída com sucesso!")


if __name__ == "__main__":
    main()
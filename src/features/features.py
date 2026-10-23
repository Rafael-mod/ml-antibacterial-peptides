"""
src/features/build_features.py

Engenharia de Features para Peptídeos Antimicrobianos:
1. Extrai propriedades físico-químicas, AAC, Z-scales, PC6 e PseAAC.
2. Gera One-Hot Encoding posicional com padding.
3. Salva 'dataset_unificado.csv' e 'dataset_treino.csv' em 'data/processed/'.

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

# Hellberg Z-scales (z1 a z5)
Z_SCALES = {
    'A': [0.24, -2.32, 0.60, -0.14, 1.30], 'C': [0.84, -1.67, 3.71, 0.18, -2.65],
    'D': [3.98, 0.93, 1.93, -2.46, 0.75], 'E': [3.11, 0.26, -0.11, -3.04, -0.25],
    'F': [-4.22, 1.94, 1.06, 0.54, -0.62], 'G': [2.05, -4.06, 0.36, -0.82, -0.38],
    'H': [2.47, 1.95, 0.26, 3.90, 0.09], 'I': [-3.89, -1.73, -1.71, -0.84, 0.26],
    'K': [2.29, 0.89, -2.49, 1.49, 0.31], 'L': [-4.28, -1.31, -1.49, -0.72, 0.84],
    'M': [-2.85, -0.22, 0.47, 1.94, -0.98], 'N': [3.05, 1.62, 1.04, -1.15, 1.61],
    'P': [-1.66, 0.27, 1.84, 0.70, 2.00], 'Q': [1.75, 0.50, -1.44, -1.34, 0.66],
    'R': [3.52, 2.50, -3.50, 1.99, -0.17], 'S': [2.39, -1.07, 1.15, -1.39, 0.67],
    'T': [0.75, -2.18, -1.12, -1.46, -0.40], 'V': [-2.59, -2.64, -1.54, -0.85, -0.02],
    'W': [-4.36, 3.94, 0.59, 3.44, -1.59], 'Y': [-2.54, 2.44, 0.43, 0.04, -1.47]
}

# PC6 (6 physicochemical properties)
PC6 = {
    'A': [0.008, 0.134, -0.475, -0.039, 0.181, 0.047], 'C': [-0.166, -0.162, 0.027, 1.258, 0.164, 1.206],
    'D': [1.050, 0.302, -3.656, -0.259, -3.242, 2.046], 'E': [1.357, -1.453, 1.477, 0.113, -0.837, -1.579],
    'F': [-1.006, -0.590, 1.891, -0.397, 0.412, -0.021], 'G': [-0.417, 1.343, -1.045, 1.026, 0.957, -0.161],
    'H': [0.336, -0.417, -1.673, -1.474, -0.078, -0.042], 'I': [-1.239, -0.547, 2.131, 0.393, 0.816, 0.040],
    'K': [1.831, -0.561, 0.533, -0.277, 1.648, -0.063], 'L': [-1.019, -0.987, -1.505, 1.266, -0.912, 0.047],
    'M': [-0.663, -1.524, 2.219, -1.005, 1.212, 0.052], 'N': [0.945, 0.828, 1.299, -0.169, 0.933, -0.902],
    'P': [0.189, 2.081, -1.628, 0.421, -1.392, -0.360], 'Q': [0.931, -0.179, -3.005, -0.503, -1.853, -0.667],
    'R': [1.538, -0.055, 1.502, 0.440, 2.897, -0.062], 'S': [0.852, 0.226, -0.129, 0.502, 0.726, 0.054],
    'T': [0.652, 0.010, -0.049, -1.189, 0.323, 0.050], 'V': [-0.912, -0.247, -2.570, 0.207, 0.692, 0.046],
    'W': [-0.595, -0.009, 0.672, -2.128, -0.184, -0.053], 'Y': [0.260, 0.830, 3.097, -0.838, -1.412, -0.045]
}

# ==========================================
# Funções Auxiliares
# ==========================================
def _validar(seq: str) -> str:
    seq = str(seq).strip().upper()
    invalidos = set(seq) - set(AMINOACIDOS)
    if not seq or invalidos:
        raise ValueError(f"Sequência inválida ({seq!r})")
    return seq

def _momento_hidrofobico(seq: str, angulo_graus: float = 100.0) -> float:
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
        contagem = ProteinAnalysis(_validar(seq)).count_amino_acids()
        linhas.append({f"AAC_{aa}": contagem[aa] / len(seq) for aa in AMINOACIDOS})
    return pd.DataFrame(linhas, index=df.index)

def calcular_zscales_pc6(df: pd.DataFrame, col_seq: str = "sequence") -> pd.DataFrame:
    linhas = []
    for seq in df[col_seq]:
        seq = _validar(seq)
        z = np.array([Z_SCALES[aa] for aa in seq])
        p = np.array([PC6[aa] for aa in seq])
        linhas.append({
            **{f"Zscale_mean_{i+1}": m for i, m in enumerate(z.mean(axis=0))},
            **{f"PC6_mean_{i+1}": m for i, m in enumerate(p.mean(axis=0))},
        })
    return pd.DataFrame(linhas, index=df.index)

def calcular_pseaac_simplificado(df: pd.DataFrame, col_seq: str = "sequence", max_lambda: int = 4) -> pd.DataFrame:
    """Calcula componentes de correlação de ordem de sequência (pseudo-components)."""
    linhas = []
    for seq in df[col_seq]:
        seq = _validar(seq)
        L = len(seq)
        features = {}
        h = np.array([EISENBERG[aa] for aa in seq]) # Usando hidrofobicidade como base
        
        atual_lambda = min(max_lambda, L - 2) if L > 2 else 0
        for lam in range(1, atual_lambda + 1):
            theta = sum((h[i] - h[i + lam])**2 for i in range(L - lam)) / (L - lam)
            features[f"PseAAC_lambda_{lam}"] = theta
            
        # Preenche com 0 caso a sequência seja muito curta
        for lam in range(atual_lambda + 1, max_lambda + 1):
            features[f"PseAAC_lambda_{lam}"] = 0.0
            
        linhas.append(features)
    return pd.DataFrame(linhas, index=df.index)

def one_hot_encoding(df: pd.DataFrame, col_seq: str = "sequence", max_len: int = 50) -> pd.DataFrame:
    indice_aa = {aa: i for i, aa in enumerate(AMINOACIDOS)}
    seqs = [_validar(s) for s in df[col_seq]]
    matriz = np.zeros((len(seqs), max_len, len(AMINOACIDOS)), dtype=np.uint8)
    for i, seq in enumerate(seqs):
        for j, aa in enumerate(seq[:max_len]): # Trunca se for maior
            matriz[i, j, indice_aa[aa]] = 1
    colunas = [f"OH_p{p + 1:02d}_{aa}" for p in range(max_len) for aa in AMINOACIDOS]
    return pd.DataFrame(matriz.reshape(len(seqs), -1), columns=colunas, index=df.index)

# ==========================================
# Montagem e Execução
# ==========================================
def construir_features(seed: int = 42):
    print("=== [1/3] Carregando Datasets ===")
    pos = pd.read_csv(CSV_POS)
    neg = pd.read_csv(CSV_NEG)
    if "group" not in neg.columns: neg["group"] = "negativo"
    
    unificado = pd.concat([pos, neg], ignore_index=True).drop_duplicates(subset="sequence")
    unificado = unificado.sample(frac=1, random_state=seed).reset_index(drop=True)

    print("\n=== [2/3] Extraindo Todas as Features Brutas ===")
    print("  -> Propriedades físico-químicas...")
    df_prop = calcular_propriedades_fisicoquimicas(unificado)
    
    print("  -> AAC (Composição de Aminoácidos)...")
    df_aac = calcular_aac(unificado)
    
    print("  -> Z-scales e Matrizes PC6...")
    df_z_pc6 = calcular_zscales_pc6(unificado)
    
    print("  -> PseAAC (Correlação de Sequência)...")
    df_pseaac = calcular_pseaac_simplificado(unificado)
    
    print("  -> One-Hot Encoding...")
    df_ohe = one_hot_encoding(unificado)

    # Junta todas as features geradas sem qualquer filtro
    X_todas = pd.concat([df_prop, df_aac, df_z_pc6, df_pseaac, df_ohe], axis=1)

    treino_final = pd.concat([unificado[["id", "sequence", "label"]], X_todas], axis=1)
    return unificado, treino_final

def main():
    unificado, treino = construir_features()
    PROC.mkdir(parents=True, exist_ok=True)
    unificado.to_csv(OUT_UNIFICADO, index=False)
    treino.to_csv(OUT_TREINO, index=False)
    print("\n=== [3/3] Arquivos Salvos ===")
    print(f"  -> Treino Bruto gerado: {OUT_TREINO.name} | Total de Features: {treino.shape[1] - 3}")

if __name__ == "__main__":
    main()
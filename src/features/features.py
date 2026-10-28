"""
src/features/build_features.py

Engenharia de Features para Peptídeos Antimicrobianos.
Extrai a máxima riqueza de sinal biológico sem vazamento de dados, mapeando a
sequência em um espaço vetorial numérico de dimensão fixa.

TODAS AS FEATURES CALCULADAS NESTE ARQUIVO:
=============================================================================
1. Físico-Químicas Globais: Comprimento, Peso Molecular, Carga (pH 7.4), pI, 
   GRAVY, Índice de Boman, Aromaticidade, Instabilidade, Índice Alifático, 
   Frações (Hélice/Volta/Folha do ProtParam, Polar, Hidrofóbico, etc.).
2. Momento Hidrofóbico Estendido (μH): hélice (100°) e folha-beta (180°).
3. Propensão Conformacional (Chou-Fasman): Hélice, Folha e Alça/Turn.
4. Acessibilidade ao Solvente: Fração exposta, intermediária e enterrada.
5. AAC (Composição de Aminoácidos): 20 features.
6. DPC (Dipeptide Composition): 400 features (AA, AC, ..., YY).
7. CTC (Conjoint Triad Descriptors): 343 features baseadas em 7 grupos químicos.
8. CKSAAP (Composição de Pares Espaçados): 1600 features (k=1 a 4).
9. ATC (Composição Atômica): 5 features (C, H, N, O, S normalizados).
10. Z-scales (Hellberg): 5 médias de descritores latentes.
11. PC6: 6 propriedades físico-químicas latentes médias.
12. PseAAC (Pseudo-Amino Acid Composition): 4 features de ordem de sequência.
13. One-Hot Encoding: Representação posicional truncada c/ padding (ex: 50x20).
=============================================================================

Uso:
    python src/features/build_features.py
"""

import itertools
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

# Índice de Boman (Estimativa de afinidade por proteínas)
BOMAN = {
    "L": -4.92, "I": -4.92, "V": -4.04, "F": -2.98, "M": -2.35, "W": -2.33,
    "A": -1.81, "C": -1.28, "G": -0.94, "Y": 0.14, "T": 2.50, "S": 3.40,
    "P": 3.14, "H": 3.30, "Q": 5.54, "N": 6.64, "E": 8.43, "D": 8.60,
    "K": 14.92, "R": 14.92
}

# Propensões Chou-Fasman (Hélice, Folha, Alça)
CHOU_FASMAN = {
    'A': (1.42, 0.83, 0.66), 'R': (0.98, 0.93, 0.95), 'N': (0.67, 0.89, 1.56),
    'D': (1.01, 0.54, 1.46), 'C': (0.70, 1.19, 1.19), 'Q': (1.11, 1.10, 0.98),
    'E': (1.51, 0.37, 0.74), 'G': (0.57, 0.75, 1.56), 'H': (1.00, 0.87, 0.95),
    'I': (1.08, 1.60, 0.47), 'L': (1.21, 1.30, 0.59), 'K': (1.16, 0.74, 1.01),
    'M': (1.45, 1.05, 0.60), 'F': (1.13, 1.38, 0.60), 'P': (0.57, 0.55, 1.52),
    'S': (0.77, 0.75, 1.43), 'T': (0.83, 1.19, 0.96), 'W': (1.08, 1.37, 0.96),
    'Y': (0.69, 1.47, 1.14), 'V': (1.06, 1.70, 0.50)
}

# Grupos de Acessibilidade ao Solvente
BURIED = set("CILMFVW")
INTERMEDIATE = set("AGHPSTY")
EXPOSED = set("RNDQEK")

# Mapeamento CTC (Conjoint Triad) - 7 Classes Físico-Químicas
CTC_MAP = {
    'A': '1', 'G': '1', 'V': '1',
    'I': '2', 'L': '2', 'F': '2', 'P': '2',
    'Y': '3', 'M': '3', 'T': '3', 'S': '3',
    'H': '4', 'N': '4', 'Q': '4', 'W': '4',
    'R': '5', 'K': '5',
    'D': '6', 'E': '6',
    'C': '7'
}

# Composição Atômica dos Resíduos (Descontando H2O da ligação peptídica)
# Ordem: [Carbono, Hidrogênio, Nitrogênio, Oxigênio, Enxofre]
ATOMIC_COMP = {
    'A': [3, 5, 1, 1, 0], 'R': [6, 12, 4, 1, 0], 'N': [4, 6, 2, 2, 0],
    'D': [4, 5, 1, 3, 0], 'C': [3, 5, 1, 1, 1],  'Q': [5, 8, 2, 2, 0],
    'E': [5, 7, 1, 3, 0], 'G': [2, 3, 1, 1, 0],  'H': [6, 7, 3, 1, 0],
    'I': [6, 11, 1, 1, 0], 'L': [6, 11, 1, 1, 0], 'K': [6, 12, 2, 1, 0],
    'M': [5, 9, 1, 1, 1],  'F': [9, 9, 1, 1, 0],  'P': [5, 7, 1, 1, 0],
    'S': [3, 5, 1, 2, 0],  'T': [4, 7, 1, 2, 0],  'W': [11, 10, 2, 1, 0],
    'Y': [9, 9, 1, 2, 0],  'V': [5, 9, 1, 1, 0]
}

# Hellberg Z-scales
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

# PC6 (6 propriedades)
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
# Funções Auxiliares de Tratamento
# ==========================================
def _validar(seq: str) -> str:
    seq = str(seq).strip().upper()
    invalidos = set(seq) - set(AMINOACIDOS)
    if not seq or invalidos:
        # Se houver resíduos não canônicos, remove ou trata (aqui substituímos/filtramos)
        seq = ''.join([aa for aa in seq if aa in AMINOACIDOS])
    return seq

def _momento_hidrofobico(seq: str, angulo_graus: float) -> float:
    if not seq: return 0.0
    ang = np.deg2rad(angulo_graus) * np.arange(len(seq))
    h = np.array([EISENBERG[aa] for aa in seq])
    return float(np.hypot((h * np.cos(ang)).sum(), (h * np.sin(ang)).sum()) / len(seq))


# ==========================================
# 1. Extrator Global (Físico-Químico, Propensões, ATC)
# ==========================================
def calcular_features_globais(df: pd.DataFrame, col_seq: str = "sequence", ph: float = 7.4) -> pd.DataFrame:
    linhas = []
    for seq in df[col_seq]:
        seq = _validar(seq)
        n = len(seq)
        if n == 0:
            continue
            
        pa = ProteinAnalysis(seq)
        contagem = pa.count_amino_acids()
        
        # 1. Físico-Químicas Biopython
        pct = lambda aa: 100.0 * contagem[aa] / n
        frac = lambda conjunto: sum(contagem[aa] for aa in conjunto) / n
        
        boman_idx = sum(BOMAN[aa] for aa in seq) / n
        
        # 2. Propensão Conformacional Chou-Fasman
        prop_h = sum(CHOU_FASMAN[aa][0] for aa in seq) / n
        prop_f = sum(CHOU_FASMAN[aa][1] for aa in seq) / n
        prop_a = sum(CHOU_FASMAN[aa][2] for aa in seq) / n
        
        # 3. Composição Atômica (ATC)
        atomos = np.sum([ATOMIC_COMP[aa] for aa in seq], axis=0) / n
        
        linhas.append({
            "comprimento": n,
            "peso_molecular": pa.molecular_weight(),
            "carga_liquida": pa.charge_at_pH(ph),
            "ponto_isoeletrico": pa.isoelectric_point(),
            "gravy": pa.gravy(),
            "indice_boman": boman_idx,
            "aromaticidade": pa.aromaticity(),
            "indice_instabilidade": pa.instability_index(),
            "indice_alifatico": pct("A") + 2.9 * pct("V") + 3.9 * (pct("I") + pct("L")),
            "frac_hidrofobica": frac(BURIED),
            "frac_polar": frac(EXPOSED),
            "momento_hidrofobico_helice": _momento_hidrofobico(seq, 100.0),
            "momento_hidrofobico_folha": _momento_hidrofobico(seq, 180.0),
            "propensao_helice": prop_h,
            "propensao_folha": prop_f,
            "propensao_alca": prop_a,
            "fracao_exposta_solvente": frac(EXPOSED),
            "fracao_enterrada_membrana": frac(BURIED),
            "ATC_C": atomos[0], "ATC_H": atomos[1], "ATC_N": atomos[2], "ATC_O": atomos[3], "ATC_S": atomos[4]
        })
    return pd.DataFrame(linhas, index=df.index)

# ==========================================
# 2. Extratores de Composição (AAC, DPC, CTC, CKSAAP)
# ==========================================
def calcular_aac(df: pd.DataFrame, col_seq: str = "sequence") -> pd.DataFrame:
    linhas = []
    for seq in df[col_seq]:
        seq = _validar(seq)
        contagem = ProteinAnalysis(seq).count_amino_acids()
        linhas.append({f"AAC_{aa}": contagem[aa] / len(seq) for aa in AMINOACIDOS})
    return pd.DataFrame(linhas, index=df.index)

def calcular_dpc(df: pd.DataFrame, col_seq: str = "sequence") -> pd.DataFrame:
    pares = ["".join(p) for p in itertools.product(AMINOACIDOS, repeat=2)]
    linhas = []
    for seq in df[col_seq]:
        seq = _validar(seq)
        n = len(seq)
        if n < 2:
            linhas.append({f"DPC_{p}": 0.0 for p in pares})
            continue
        
        contagem = {p: 0 for p in pares}
        for i in range(n - 1):
            par = seq[i:i+2]
            if par in contagem: contagem[par] += 1
            
        linhas.append({f"DPC_{p}": val / (n - 1) for p, val in contagem.items()})
    return pd.DataFrame(linhas, index=df.index)

def calcular_ctc(df: pd.DataFrame, col_seq: str = "sequence") -> pd.DataFrame:
    classes = ["1", "2", "3", "4", "5", "6", "7"]
    trincas = ["".join(t) for t in itertools.product(classes, repeat=3)]
    linhas = []
    
    for seq in df[col_seq]:
        seq = _validar(seq)
        n = len(seq)
        if n < 3:
            linhas.append({f"CTC_{t}": 0.0 for t in trincas})
            continue
            
        seq_grupos = "".join([CTC_MAP[aa] for aa in seq])
        contagem = {t: 0 for t in trincas}
        for i in range(n - 2):
            trinca = seq_grupos[i:i+3]
            contagem[trinca] += 1
            
        linhas.append({f"CTC_{t}": val / (n - 2) for t, val in contagem.items()})
    return pd.DataFrame(linhas, index=df.index)

def calcular_cksaap(df: pd.DataFrame, col_seq: str = "sequence") -> pd.DataFrame:
    pares = ["".join(p) for p in itertools.product(AMINOACIDOS, repeat=2)]
    linhas = []
    
    for seq in df[col_seq]:
        seq = _validar(seq)
        n = len(seq)
        features = {}
        
        for k in range(1, 5):  # k = 1, 2, 3, 4
            contagem = {p: 0 for p in pares}
            total_pares = n - k - 1
            
            if total_pares > 0:
                for i in range(total_pares):
                    par = seq[i] + seq[i + k + 1]
                    if par in contagem: contagem[par] += 1
                for p in pares:
                    features[f"CKSAAP_k{k}_{p}"] = contagem[p] / total_pares
            else:
                for p in pares:
                    features[f"CKSAAP_k{k}_{p}"] = 0.0
                    
        linhas.append(features)
    return pd.DataFrame(linhas, index=df.index)

# ==========================================
# 3. Extratores Latentes e Posicionais
# ==========================================
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

def calcular_pseaac(df: pd.DataFrame, col_seq: str = "sequence", max_lambda: int = 4) -> pd.DataFrame:
    linhas = []
    for seq in df[col_seq]:
        seq = _validar(seq)
        L = len(seq)
        features = {}
        h = np.array([EISENBERG[aa] for aa in seq]) 
        
        atual_lambda = min(max_lambda, L - 2) if L > 2 else 0
        for lam in range(1, atual_lambda + 1):
            theta = sum((h[i] - h[i + lam])**2 for i in range(L - lam)) / (L - lam)
            features[f"PseAAC_lambda_{lam}"] = theta
            
        for lam in range(atual_lambda + 1, max_lambda + 1):
            features[f"PseAAC_lambda_{lam}"] = 0.0
            
        linhas.append(features)
    return pd.DataFrame(linhas, index=df.index)

def one_hot_encoding(df: pd.DataFrame, col_seq: str = "sequence", max_len: int = 50) -> pd.DataFrame:
    indice_aa = {aa: i for i, aa in enumerate(AMINOACIDOS)}
    seqs = [_validar(s) for s in df[col_seq]]
    matriz = np.zeros((len(seqs), max_len, len(AMINOACIDOS)), dtype=np.uint8)
    
    for i, seq in enumerate(seqs):
        for j, aa in enumerate(seq[:max_len]): 
            matriz[i, j, indice_aa[aa]] = 1
            
    colunas = [f"OH_p{p + 1:02d}_{aa}" for p in range(max_len) for aa in AMINOACIDOS]
    return pd.DataFrame(matriz.reshape(len(seqs), -1), columns=colunas, index=df.index)

# ==========================================
# Montagem Principal
# ==========================================
def construir_features(seed: int = 42):
    print("=== [1/3] Carregando Datasets ===")
    pos = pd.read_csv(CSV_POS)
    neg = pd.read_csv(CSV_NEG)
    if "group" not in neg.columns: neg["group"] = "negativo"
    
    unificado = pd.concat([pos, neg], ignore_index=True).drop_duplicates(subset="sequence")
    unificado = unificado.sample(frac=1, random_state=seed).reset_index(drop=True)

    print("\n=== [2/3] Extraindo Todas as Features ===")
    
    print("  -> Extraindo Propriedades Físico-Químicas Globais, Propensão Conformacional e ATC...")
    df_prop = calcular_features_globais(unificado)
    
    print("  -> Extraindo AAC (Composição de Aminoácidos: 20)...")
    df_aac = calcular_aac(unificado)
    
    print("  -> Extraindo DPC (Composição de Dipeptídeos: 400)...")
    df_dpc = calcular_dpc(unificado)
    
    print("  -> Extraindo CTC (Triplas Conjuntas Físico-Químicas: 343)...")
    df_ctc = calcular_ctc(unificado)
    
    print("  -> Extraindo CKSAAP (Pares Espaçados k=1..4: 1600)...")
    df_cksaap = calcular_cksaap(unificado)
    
    print("  -> Extraindo Descritores Latentes Z-scales e PC6...")
    df_z_pc6 = calcular_zscales_pc6(unificado)
    
    print("  -> Extraindo PseAAC (Correlação de Sequência)...")
    df_pseaac = calcular_pseaac(unificado)
    
    print("  -> Extraindo One-Hot Encoding...")
    df_ohe = one_hot_encoding(unificado)

    # Concatenação final
    print("\n  -> Concatenando matriz final...")
    X_todas = pd.concat([df_prop, df_aac, df_dpc, df_ctc, df_cksaap, df_z_pc6, df_pseaac, df_ohe], axis=1)
    treino_final = pd.concat([unificado[["id", "sequence", "label"]], X_todas], axis=1)
    
    return unificado, treino_final

def main():
    unificado, treino = construir_features()
    PROC.mkdir(parents=True, exist_ok=True)
    unificado.to_csv(OUT_UNIFICADO, index=False)
    treino.to_csv(OUT_TREINO, index=False)
    print("\n=== [3/3] Arquivos Salvos com Sucesso ===")
    print(f"  -> Treino Bruto gerado: {OUT_TREINO.name}")
    print(f"  -> Total de Features extraídas por peptídeo: {treino.shape[1] - 3}")

if __name__ == "__main__":
    main()
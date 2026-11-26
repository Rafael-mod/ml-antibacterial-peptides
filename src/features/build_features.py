"""
src/features/build_features.py

Engenharia de Features para Peptídeos Antimicrobianos (ABPs):
1. Extrai propriedades físico-químicas (incl. Boman, momento hidrofóbico alfa/beta,
   densidade de carga e resumo dos terminais N5/C5), AAC, Z-scales, PC6 e
   Correlação de Ordem de Sequência na escala de Eisenberg (PseAAC simplificado).
2. Opcionalmente gera One-Hot Encoding corrigido para terminais N5 e C5
   (OHE_C1 = seq[-1], ..., OHE_C5 = seq[-5]).
3. Preserva metadados críticos ('id', 'sequence', 'label', 'cluster_id', 'group')
   e expõe 'extrair_features_dataframe' para reuso em benchmarks externos (ex.: PGAT-ABPp).
"""

from pathlib import Path
import numpy as np
import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis

# ==========================================
# Configurações e Diretórios do Projeto
# ==========================================
ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"

CSV_POS = PROC / "dataset_positivo.csv"
CSV_NEG = PROC / "dataset_negativo.csv"

OUT_UNIFICADO = PROC / "dataset_unificado.csv"
OUT_TREINO = PROC / "dataset_treino.csv"

# Flag para teste de ablação (200 colunas esparsas podem gerar ruído em split 40%)
INCLUIR_OHE_TERMINAIS = True
K_TERMINAL = 5

# ==========================================
# Constantes Biológicas e Químicas
# ==========================================
AMINOACIDOS = list("ACDEFGHIKLMNPQRSTVWY")
SET_AMINOACIDOS = set(AMINOACIDOS)

# Escala de hidrofobicidade consenso de Eisenberg
EISENBERG = {
    "A": 0.62, "R": -2.53, "N": -0.78, "D": -0.90, "C": 0.29, "Q": -0.85,
    "E": -0.74, "G": 0.48, "H": -0.40, "I": 1.38, "L": 1.06, "K": -1.50,
    "M": 0.64, "F": 1.19, "P": 0.12, "S": -0.18, "T": -0.05, "W": 0.81,
    "Y": 0.26, "V": 1.08,
}

# Escala de Boman (2003) - potencial de ligação a proteínas/membranas (kcal/mol)
BOMAN = {
    "L": -4.92, "I": -4.92, "V": -4.04, "F": -2.98, "M": -2.35, "W": -2.33,
    "A": -1.81, "C": -1.28, "G": -0.94, "Y": 0.14,  "T": 2.57,  "S": 3.40,
    "H": 4.66,  "Q": 5.54,  "K": 5.55,  "N": 6.64,  "E": 6.81,  "D": 8.72,
    "R": 14.92, "P": 0.00,
}

HIDROFOBICOS = set("AILMFVW")
POLARES = set("STNQ")
POSITIVOS = set("KRH")
NEGATIVOS = set("DE")

HELICE_FORMADORES = set("ALMKRQEI")
HELICE_QUEBRADORES = set("PG")

# Hellberg Z-scales (z1 a z5)
Z_SCALES = {
    "A": [0.24, -2.32, 0.60, -0.14, 1.30],  "C": [0.84, -1.67, 3.71, 0.18, -2.65],
    "D": [3.98, 0.93, 1.93, -2.46, 0.75],   "E": [3.11, 0.26, -0.11, -3.04, -0.25],
    "F": [-4.22, 1.94, 1.06, 0.54, -0.62],  "G": [2.05, -4.06, 0.36, -0.82, -0.38],
    "H": [2.47, 1.95, 0.26, 3.90, 0.09],    "I": [-3.89, -1.73, -1.71, -0.84, 0.26],
    "K": [2.29, 0.89, -2.49, 1.49, 0.31],   "L": [-4.28, -1.31, -1.49, -0.72, 0.84],
    "M": [-2.85, -0.22, 0.47, 1.94, -0.98], "N": [3.05, 1.62, 1.04, -1.15, 1.61],
    "P": [-1.66, 0.27, 1.84, 0.70, 2.00],   "Q": [1.75, 0.50, -1.44, -1.34, 0.66],
    "R": [3.52, 2.50, -3.50, 1.99, -0.17],  "S": [2.39, -1.07, 1.15, -1.39, 0.67],
    "T": [0.75, -2.18, -1.12, -1.46, -0.40],"V": [-2.59, -2.64, -1.54, -0.85, -0.02],
    "W": [-4.36, 3.94, 0.59, 3.44, -1.59],  "Y": [-2.54, 2.44, 0.43, 0.04, -1.47],
}

# PC6 (6 propriedades físico-químicas projetadas)
PC6 = {
    "A": [0.008, 0.134, -0.475, -0.039, 0.181, 0.047],  "C": [-0.166, -0.162, 0.027, 1.258, 0.164, 1.206],
    "D": [1.050, 0.302, -3.656, -0.259, -3.242, 2.046], "E": [1.357, -1.453, 1.477, 0.113, -0.837, -1.579],
    "F": [-1.006, -0.590, 1.891, -0.397, 0.412, -0.021],"G": [-0.417, 1.343, -1.045, 1.026, 0.957, -0.161],
    "H": [0.336, -0.417, -1.673, -1.474, -0.078, -0.042],"I": [-1.239, -0.547, 2.131, 0.393, 0.816, 0.040],
    "K": [1.831, -0.561, 0.533, -0.277, 1.648, -0.063], "L": [-1.019, -0.987, -1.505, 1.266, -0.912, 0.047],
    "M": [-0.663, -1.524, 2.219, -1.005, 1.212, 0.052], "N": [0.945, 0.828, 1.299, -0.169, 0.933, -0.902],
    "P": [0.189, 2.081, -1.628, 0.421, -1.392, -0.360], "Q": [0.931, -0.179, -3.005, -0.503, -1.853, -0.667],
    "R": [1.538, -0.055, 1.502, 0.440, 2.897, -0.062],  "S": [0.852, 0.226, -0.129, 0.502, 0.726, 0.054],
    "T": [0.652, 0.010, -0.049, -1.189, 0.323, 0.050],  "V": [-0.912, -0.247, -2.570, 0.207, 0.692, 0.046],
    "W": [-0.595, -0.009, 0.672, -2.128, -0.184, -0.053],"Y": [0.260, 0.830, 3.097, -0.838, -1.412, -0.045],
}


# ==========================================
# Funções Auxiliares
# ==========================================
def _validar(seq: str) -> str:
    seq = str(seq).strip().upper()
    invalidos = set(seq) - SET_AMINOACIDOS
    if not seq or invalidos:
        raise ValueError(f"Sequência inválida ({seq!r}): resíduos {invalidos}")
    return seq


def _momento_hidrofobico(h: np.ndarray, angulo_graus: float) -> float:
    ang = np.deg2rad(angulo_graus) * np.arange(len(h))
    return float(np.hypot((h * np.cos(ang)).sum(), (h * np.sin(ang)).sum()) / len(h))


def _normalizar_coluna_cluster(df: pd.DataFrame) -> pd.DataFrame:
    """Padroniza o nome da coluna de cluster 40% para 'cluster_id'."""
    candidatas = ["cluster_id", "cluster_40", "cluster40", "cluster"]
    for col in candidatas:
        if col in df.columns:
            if col != "cluster_id":
                df = df.rename(columns={col: "cluster_id"})
            return df
    raise KeyError(
        f"Coluna de cluster 40% não encontrada (esperado uma de {candidatas}). "
        f"Colunas disponíveis: {list(df.columns)}"
    )


# ==========================================
# Calculadores de Features
# ==========================================
def calcular_propriedades_fisicoquimicas(
    df: pd.DataFrame, col_seq: str = "sequence", ph: float = 7.4, k_term: int = K_TERMINAL
) -> pd.DataFrame:
    linhas = []
    for seq in df[col_seq]:
        seq = _validar(seq)
        n = len(seq)
        pa = ProteinAnalysis(seq)
        contagem = pa.count_amino_acids()
        helice, _, folha = pa.secondary_structure_fraction()

        h = np.array([EISENBERG[aa] for aa in seq], dtype=np.float64)
        carga = pa.charge_at_pH(ph)
        boman_idx = float(sum(BOMAN[aa] for aa in seq) / n)

        # Resumo denso dos terminais N5 e C5 (captura sinal terminal sem esparsidade)
        n_term = seq[:k_term]
        c_term = seq[-k_term:]
        carga_n = ProteinAnalysis(n_term).charge_at_pH(ph)
        carga_c = ProteinAnalysis(c_term).charge_at_pH(ph)
        hidrofob_n = float(np.mean([EISENBERG[aa] for aa in n_term]))
        hidrofob_c = float(np.mean([EISENBERG[aa] for aa in c_term]))

        pct = lambda aa: 100.0 * contagem[aa] / n
        frac = lambda conjunto: sum(contagem[aa] for aa in conjunto) / n

        linhas.append({
            "comprimento": n,
            "peso_molecular": pa.molecular_weight(),
            "carga_liquida": carga,
            "densidade_carga": carga / n,
            "ponto_isoeletrico": pa.isoelectric_point(),
            "gravy": pa.gravy(),
            "indice_boman": boman_idx,
            "aromaticidade": pa.aromaticity(),
            "indice_instabilidade": pa.instability_index(),
            "indice_alifatico": pct("A") + 2.9 * pct("V") + 3.9 * (pct("I") + pct("L")),
            "frac_helice_biopython": helice,
            "frac_folha_biopython": folha,
            "frac_hidrofobica": frac(HIDROFOBICOS),
            "frac_polar": frac(POLARES),
            "frac_positiva": frac(POSITIVOS),
            "frac_negativa": frac(NEGATIVOS),
            "propensao_helice_alfa": frac(HELICE_FORMADORES),
            "quebradores_helice": frac(HELICE_QUEBRADORES),
            "momento_hidrofobico_alfa": _momento_hidrofobico(h, 100.0),
            "momento_hidrofobico_beta": _momento_hidrofobico(h, 160.0),
            "carga_N5": carga_n,
            "carga_C5": carga_c,
            "hidrofob_N5": hidrofob_n,
            "hidrofob_C5": hidrofob_c,
        })
    return pd.DataFrame(linhas, index=df.index)


def calcular_aac(df: pd.DataFrame, col_seq: str = "sequence") -> pd.DataFrame:
    linhas = []
    for seq in df[col_seq]:
        seq = _validar(seq)
        n = len(seq)
        contagem = ProteinAnalysis(seq).count_amino_acids()
        linhas.append({f"AAC_{aa}": contagem[aa] / n for aa in AMINOACIDOS})
    return pd.DataFrame(linhas, index=df.index)


def calcular_zscales_pc6(df: pd.DataFrame, col_seq: str = "sequence") -> pd.DataFrame:
    linhas = []
    for seq in df[col_seq]:
        seq = _validar(seq)
        z = np.array([Z_SCALES[aa] for aa in seq], dtype=np.float64)
        p = np.array([PC6[aa] for aa in seq], dtype=np.float64)
        linhas.append({
            **{f"Zscale_mean_{i + 1}": float(m) for i, m in enumerate(z.mean(axis=0))},
            **{f"PC6_mean_{i + 1}": float(m) for i, m in enumerate(p.mean(axis=0))},
        })
    return pd.DataFrame(linhas, index=df.index)


def calcular_soc_eisenberg(
    df: pd.DataFrame, col_seq: str = "sequence", max_lambda: int = 4
) -> pd.DataFrame:
    """
    Correlação de Ordem de Sequência (SOC) baseada na hidrofobicidade de Eisenberg
    (PseAAC simplificado sem normalização de Chou, 2001).
    """
    linhas = []
    for seq in df[col_seq]:
        seq = _validar(seq)
        h = np.array([EISENBERG[aa] for aa in seq], dtype=np.float64)
        L = len(h)
        features = {}

        atual_lambda = min(max_lambda, L - 2) if L > 2 else 0
        for lam in range(1, atual_lambda + 1):
            diff = h[:-lam] - h[lam:]
            features[f"SOC_Eisenberg_lambda_{lam}"] = float(np.mean(diff * diff))

        for lam in range(atual_lambda + 1, max_lambda + 1):
            features[f"SOC_Eisenberg_lambda_{lam}"] = 0.0

        linhas.append(features)
    return pd.DataFrame(linhas, index=df.index)


def one_hot_encoding_terminais(
    df: pd.DataFrame, col_seq: str = "sequence", k: int = K_TERMINAL
) -> pd.DataFrame:
    """
    One-Hot Encoding estrito dos K primeiros e K últimos resíduos:
      - OHE_N1 = seq[0], ..., OHE_Nk = seq[k-1]
      - OHE_C1 = seq[-1] (C-terminal real), ..., OHE_Ck = seq[-k]
    """
    indice_aa = {aa: i for i, aa in enumerate(AMINOACIDOS)}
    seqs = [_validar(s) for s in df[col_seq]]

    matriz_n = np.zeros((len(seqs), k, len(AMINOACIDOS)), dtype=np.uint8)
    matriz_c = np.zeros((len(seqs), k, len(AMINOACIDOS)), dtype=np.uint8)

    for i, seq in enumerate(seqs):
        n_term = seq[:k]
        c_term = seq[-k:][::-1]  # Invertido para C1 ser o resíduo C-terminal (seq[-1])

        for j, aa in enumerate(n_term):
            matriz_n[i, j, indice_aa[aa]] = 1

        for j, aa in enumerate(c_term):
            matriz_c[i, j, indice_aa[aa]] = 1

    colunas_n = [f"OHE_N{p + 1}_{aa}" for p in range(k) for aa in AMINOACIDOS]
    colunas_c = [f"OHE_C{p + 1}_{aa}" for p in range(k) for aa in AMINOACIDOS]

    df_n = pd.DataFrame(matriz_n.reshape(len(seqs), -1), columns=colunas_n, index=df.index)
    df_c = pd.DataFrame(matriz_c.reshape(len(seqs), -1), columns=colunas_c, index=df.index)
    return pd.concat([df_n, df_c], axis=1)


# ==========================================
# Interface Reutilizável (Treino + Externo)
# ==========================================
def extrair_features_dataframe(
    df: pd.DataFrame,
    col_seq: str = "sequence",
    incluir_ohe: bool = INCLUIR_OHE_TERMINAIS,
) -> pd.DataFrame:
    """Extrai o conjunto completo de features para qualquer DataFrame com sequências."""
    print("  -> Propriedades físico-químicas (incl. Boman, alfa/beta e N5/C5)...")
    df_prop = calcular_propriedades_fisicoquimicas(df, col_seq=col_seq)

    print("  -> AAC (Composição de Aminoácidos)...")
    df_aac = calcular_aac(df, col_seq=col_seq)

    print("  -> Z-scales e Matrizes PC6...")
    df_z_pc6 = calcular_zscales_pc6(df, col_seq=col_seq)

    print("  -> SOC Eisenberg (Correlação de Ordem de Sequência)...")
    df_soc = calcular_soc_eisenberg(df, col_seq=col_seq)

    blocos = [df_prop, df_aac, df_z_pc6, df_soc]

    if incluir_ohe:
        print(f"  -> One-Hot Encoding de Terminais (N{K_TERMINAL} e C{K_TERMINAL})...")
        blocos.append(one_hot_encoding_terminais(df, col_seq=col_seq, k=K_TERMINAL))
    else:
        print("  -> One-Hot Encoding de Terminais desativado (modo enxuto).")

    return pd.concat(blocos, axis=1)


# ==========================================
# Montagem e Execução
# ==========================================
def construir_features(seed: int = 42, incluir_ohe: bool = INCLUIR_OHE_TERMINAIS):
    print("=== [1/3] Carregando Datasets ===")
    pos = pd.read_csv(CSV_POS)
    neg = pd.read_csv(CSV_NEG)
    if "group" not in neg.columns:
        neg["group"] = "negativo"

    unificado = pd.concat([pos, neg], ignore_index=True).drop_duplicates(subset="sequence")
    unificado = _normalizar_coluna_cluster(unificado)
    unificado = unificado.sample(frac=1, random_state=seed).reset_index(drop=True)

    print(
        f"  -> Total unificado: {len(unificado)} sequências | "
        f"Clusters 40% únicos: {unificado['cluster_id'].nunique()}"
    )

    print("\n=== [2/3] Extraindo Features Brutas ===")
    X_todas = extrair_features_dataframe(unificado, col_seq="sequence", incluir_ohe=incluir_ohe)

    cols_meta = [c for c in ["id", "sequence", "label", "cluster_id", "group"] if c in unificado.columns]
    treino_final = pd.concat([unificado[cols_meta], X_todas], axis=1)
    return unificado, treino_final, len(cols_meta)


def main():
    unificado, treino, n_meta = construir_features()
    PROC.mkdir(parents=True, exist_ok=True)
    unificado.to_csv(OUT_UNIFICADO, index=False)
    treino.to_csv(OUT_TREINO, index=False)

    print("\n=== [3/3] Arquivos Salvos ===")
    print(f"  -> Unificado:   {OUT_UNIFICADO.name} ({unificado.shape[0]} linhas)")
    print(f"  -> Treino Bruto:{OUT_TREINO.name} | Metadados: {n_meta} | Features: {treino.shape[1] - n_meta}")


if __name__ == "__main__":
    main()
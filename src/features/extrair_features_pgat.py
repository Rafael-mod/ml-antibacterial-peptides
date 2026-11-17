"""
src/features/extrair_features_pgat.py

Extração de Descritores Físico-Químicos e Embeddings ESM-2 (650M)
para o Benchmark Externo PGAT-ABPp:
1. Lê 'data/external/pgat/pgat_dataset_unificado.csv' (5.149 sequências).
2. Extrai os descritores físico-químicos e estruturais e seleciona exatamente
   as 73 colunas presentes em 'data/processed/X_train_limpo.csv'.
3. Extrai os embeddings de 1.280 dimensões usando 'facebook/esm2_t33_650M_UR50D'
   com Mean Pooling exclusivo sobre os resíduos reais (sem <cls>, <eos> ou <pad>).
4. Valida o alinhamento linha a linha e salva 'X_pgat_fq.csv', 'X_pgat_esm2.csv'
   e 'y_pgat.csv' em 'data/external/pgat/'.

Uso:
    python src/features/extrair_features_pgat.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, EsmModel

# ==========================================
# Configurações e Caminhos
# ==========================================
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src" / "features"))

from build_features import (
    calcular_aac,
    calcular_propriedades_fisicoquimicas,
    calcular_pseaac_simplificado,
    calcular_zscales_pc6,
    one_hot_encoding_terminais,
)

PROC_DIR = ROOT / "data" / "processed"
EXT_DIR = ROOT / "data" / "external" / "pgat"

PATH_CSV_PGAT = EXT_DIR / "pgat_dataset_unificado.csv"
PATH_REF_COLS = PROC_DIR / "X_train_limpo.csv"

OUT_X_FQ = EXT_DIR / "X_pgat_fq.csv"
OUT_X_ESM = EXT_DIR / "X_pgat_esm2.csv"
OUT_Y = EXT_DIR / "y_pgat.csv"

MODELO_ESM = "facebook/esm2_t33_650M_UR50D"
BATCH_SIZE = 32


# ==========================================
# 1. Extração Físico-Química (73 Features)
# ==========================================
def extrair_features_fq(df_pgat: pd.DataFrame) -> pd.DataFrame:
    print("\n[1/2] Extraindo descritores físico-químicos e estruturais...")
    if not PATH_REF_COLS.exists():
        raise FileNotFoundError(f"Arquivo de referência não encontrado: {PATH_REF_COLS}")

    cols_treino = pd.read_csv(PATH_REF_COLS, nrows=0).columns.tolist()

    X_bruto = pd.concat(
        [
            calcular_propriedades_fisicoquimicas(df_pgat),
            calcular_aac(df_pgat),
            calcular_zscales_pc6(df_pgat),
            calcular_pseaac_simplificado(df_pgat),
            one_hot_encoding_terminais(df_pgat, k=5),
        ],
        axis=1,
    ).fillna(0)

    faltando = set(cols_treino) - set(X_bruto.columns)
    assert not faltando, f"Colunas ausentes na extração físico-química: {faltando}"

    X_alinhado = X_bruto[cols_treino].copy()
    assert not X_alinhado.isna().any().any(), "Valores NaN detectados na matriz físico-química!"
    assert len(X_alinhado) == len(df_pgat), "Divergência no número de linhas em X_pgat_fq!"

    print(f"  -> Matriz Físico-Química concluída: {X_alinhado.shape}")
    return X_alinhado


# ==========================================
# 2. Extração de Embeddings ESM-2 (1280-d)
# ==========================================
def extrair_embeddings_esm2(sequencias: list[str], batch_size: int = BATCH_SIZE) -> pd.DataFrame:
    print(f"\n[2/2] Carregando modelo {MODELO_ESM}...")
    tokenizer = AutoTokenizer.from_pretrained(MODELO_ESM)
    model = EsmModel.from_pretrained(MODELO_ESM)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  -> Dispositivo: {device} | Batch size: {batch_size} | Sequências: {len(sequencias)}")
    model = model.to(device).eval()

    embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(sequencias), batch_size), desc="Extraindo ESM-2"):
            lote = sequencias[i : i + batch_size]
            inputs = tokenizer(
                lote,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)

            for j in range(len(lote)):
                tam_real = int((inputs["attention_mask"][j] == 1).sum().item()) - 2
                if tam_real > 0:
                    emb = outputs.last_hidden_state[j, 1 : tam_real + 1, :].mean(dim=0)
                else:
                    emb = outputs.last_hidden_state[j].mean(dim=0)
                embeddings.append(emb.cpu().numpy())

    df_emb = pd.DataFrame(np.vstack(embeddings))
    assert not df_emb.isna().any().any(), "Valores NaN detectados nos embeddings ESM-2!"
    assert len(df_emb) == len(sequencias), "Divergência no número de linhas em X_pgat_esm2!"

    print(f"  -> Matriz ESM-2 concluída: {df_emb.shape}")
    return df_emb


# ==========================================
# Pipeline Principal
# ==========================================
def main() -> None:
    t0 = time.time()
    print("=== Extração de Representações — Benchmark Externo PGAT-ABPp ===")

    if not PATH_CSV_PGAT.exists():
        raise FileNotFoundError(f"Dataset não encontrado: {PATH_CSV_PGAT}")

    df_pgat = pd.read_csv(PATH_CSV_PGAT)
    print(f"Dataset carregado: {len(df_pgat)} peptídeos ({(df_pgat['label'] == 1).sum()} Pos | {(df_pgat['label'] == 0).sum()} Neg)")

    # 1. Extrai ambas as representações na mesma ordem de df_pgat
    X_pgat_fq = extrair_features_fq(df_pgat)
    X_pgat_esm = extrair_embeddings_esm2(df_pgat["sequence"].tolist())
    y_pgat = df_pgat[["label"]].copy()

    # 2. Trava final de integridade
    assert len(X_pgat_fq) == len(X_pgat_esm) == len(y_pgat), "Desalinhamento entre matrizes!"

    # 3. Exporta CSVs prontos para inferência
    X_pgat_fq.to_csv(OUT_X_FQ, index=False)
    X_pgat_esm.to_csv(OUT_X_ESM, index=False)
    y_pgat.to_csv(OUT_Y, index=False)

    print(f"\n✅ Extração concluída com sucesso em {time.time() - t0:.1f}s!")
    print(f"  -> {OUT_X_FQ.relative_to(ROOT)}   {X_pgat_fq.shape}")
    print(f"  -> {OUT_X_ESM.relative_to(ROOT)} {X_pgat_esm.shape}")
    print(f"  -> {OUT_Y.relative_to(ROOT)}      {y_pgat.shape}")


if __name__ == "__main__":
    main()
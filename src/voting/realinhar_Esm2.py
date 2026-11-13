"""
src/features/realinhar_esm2.py

Reconstrói a ordem exata dos embeddings do ESM-2 já extraídos,
alinha 1-para-1 com 'dataset_treino.csv' (mesma deduplicação e shuffle
de build_features.py) e refaz o split exato de tratamentoFeatures.py.
"""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"

SEED = 42
TEST_SIZE = 0.2


def realinhar() -> None:
    # 1. Reconstrói a tabela original usada em esm2_total_split.py
    df_pos = pd.read_csv(PROC / "dataset_positivo.csv")
    df_neg = pd.read_csv(PROC / "dataset_negativo.csv")
    df_total = pd.concat([df_pos, df_neg], ignore_index=True)
    labels_orig = df_total["label"].tolist()

    # 2. Carrega os embeddings já calculados
    X_tr_esm_salvo = pd.read_csv(PROC / "X_train_esm2.csv")
    X_te_esm_salvo = pd.read_csv(PROC / "X_test_esm2.csv")
    colunas_emb = list(X_tr_esm_salvo.columns)

    # 3. Recupera os índices exatos gerados pelo train_test_split de esm2_total_split.py
    idx_tr, idx_te = train_test_split(
        df_total.index,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=labels_orig,
    )
    X_tr_esm_salvo.index = idx_tr
    X_te_esm_salvo.index = idx_te

    # 4. Remonta a matriz completa na ordem original de df_total
    df_emb_completo = pd.concat([X_tr_esm_salvo, X_te_esm_salvo]).sort_index()
    df_emb_completo = df_emb_completo.copy()
    df_emb_completo["sequence"] = df_total["sequence"].values
    df_emb_completo["label"] = df_total["label"].values

    # 5. Aplica EXATAMENTE a mesma deduplicação e embaralhamento de build_features.py
    df_emb_alinhado = df_emb_completo.drop_duplicates(subset="sequence")
    df_emb_alinhado = df_emb_alinhado.sample(frac=1, random_state=SEED).reset_index(drop=True)

    # 6. Prova real: valida 100% das sequências contra dataset_treino.csv
    df_ref = pd.read_csv(PROC / "dataset_treino.csv")
    assert len(df_emb_alinhado) == len(df_ref), "Divergência no número de linhas!"
    assert (df_emb_alinhado["sequence"].values == df_ref["sequence"].values).all(), "Sequências desalinhadas!"
    assert (df_emb_alinhado["label"].values == df_ref["label"].values).all(), "Rótulos desalinhados!"

    # 7. Aplica EXATAMENTE o mesmo split de tratamentoFeatures.py
    X_tr_final, X_te_final, y_tr_check, y_te_check = train_test_split(
        df_emb_alinhado[colunas_emb],
        df_emb_alinhado["label"],
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=df_emb_alinhado["label"],
    )

    # 8. Confirma sincronia com y_train.csv e y_test.csv já salvos
    y_tr_disco = pd.read_csv(PROC / "y_train.csv").squeeze("columns").values
    y_te_disco = pd.read_csv(PROC / "y_test.csv").squeeze("columns").values
    assert (y_tr_check.values == y_tr_disco).all(), "y_train divergiu!"
    assert (y_te_check.values == y_te_disco).all(), "y_test divergiu!"

    # 9. Sobrescreve os CSVs agora 100% alinhados molécula-a-molécula
    X_tr_final.to_csv(PROC / "X_train_esm2.csv", index=False)
    X_te_final.to_csv(PROC / "X_test_esm2.csv", index=False)
    print("✅ Realinhamento concluído e validado molécula-a-molécula (4104 treino | 1026 teste)!")


if __name__ == "__main__":
    realinhar()
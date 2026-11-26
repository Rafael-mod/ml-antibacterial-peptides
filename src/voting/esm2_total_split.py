"""
src/features/esm2_total_split.py

Extrai embeddings por resíduo (L x 1280) do ESM-2 (650M) para toda a base
curada, gera máscaras de padding e salva tensores alinhados 1-para-1 com
X_train.csv / X_test.csv em formato PyTorch (.pt, float16) para Attention Pooling.
"""

import time
from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import AutoTokenizer, EsmModel

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"

CSV_TREINO = PROC / "dataset_treino.csv"
PATH_Y_TRAIN = PROC / "y_train.csv"
PATH_Y_TEST = PROC / "y_test.csv"

OUT_TRAIN_PT = PROC / "esm2_train_residuos.pt"
OUT_TEST_PT = PROC / "esm2_test_residuos.pt"

MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
BATCH_SIZE = 32
SEED = 42
TEST_SIZE = 0.20


def gerar_e_dividir_esm2() -> None:
    t0 = time.time()

    # 1. Carrega a base já deduplicada e embaralhada (mesma ordem de build_features.py)
    if not CSV_TREINO.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {CSV_TREINO}")

    df_ref = pd.read_csv(CSV_TREINO)
    sequencias = df_ref["sequence"].tolist()
    labels = df_ref["label"].to_numpy()
    n_total = len(sequencias)
    max_len = int(df_ref["sequence"].str.len().max())

    print(f"Base carregada: {n_total} sequências | Comprimento máximo: {max_len} AA")

    # 2. Configura o ESM-2
    print(f"Carregando modelo {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = EsmModel.from_pretrained(MODEL_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    hidden_dim = model.config.hidden_size  # 1280
    print(f"Dispositivo: {device} | Batch size: {BATCH_SIZE} | Dimensão por resíduo: {hidden_dim}")

    # 3. Pré-aloca tensores em RAM (float16 economiza 50% de memória e disco)
    emb_tensor = torch.zeros((n_total, max_len, hidden_dim), dtype=torch.float16)
    mask_tensor = torch.zeros((n_total, max_len), dtype=torch.bool)

    # 4. Extrai embeddings por resíduo em lotes (sem achatar com mean)
    print(f"\nExtraindo matrizes por resíduo para {n_total} sequências...")
    with torch.inference_mode():
        for start in tqdm(range(0, n_total, BATCH_SIZE), desc="ESM-2 Resíduos"):
            end = min(start + BATCH_SIZE, n_total)
            lote = sequencias[start:end]

            inputs = tokenizer(
                lote,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_len + 2,  # +2 para <cls> e <eos>
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)

            for j in range(len(lote)):
                # Desconta <cls> (início) e <eos> (fim)
                tam_real = int((inputs["attention_mask"][j] == 1).sum().item()) - 2
                tam_real = max(1, min(tam_real, max_len))

                # Fatia apenas os aminoácidos reais: shape (tam_real, 1280)
                res_emb = outputs.last_hidden_state[j, 1 : tam_real + 1, :]
                emb_tensor[start + j, :tam_real, :] = res_emb.to(torch.float16).cpu()
                mask_tensor[start + j, :tam_real] = True

    # 5. Divide os índices reproduzindo EXATAMENTE o split de tratamentoFeatures.py
    print("\nDividindo dados e validando alinhamento com y_train / y_test...")
    idx_tr, idx_te = train_test_split(
        df_ref.index.to_numpy(),
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=labels,
    )

    # Trava de segurança: garante sincronia 1-para-1 com as features físico-químicas
    y_tr_disco = pd.read_csv(PATH_Y_TRAIN).squeeze("columns").to_numpy()
    y_te_disco = pd.read_csv(PATH_Y_TEST).squeeze("columns").to_numpy()
    assert (labels[idx_tr] == y_tr_disco).all(), "Desalinhamento detectado em y_train!"
    assert (labels[idx_te] == y_te_disco).all(), "Desalinhamento detectado em y_test!"

    # 6. Salva pacotes completos (.pt) com embeddings, máscaras, rótulos e sequências
    pacote_treino = {
        "embeddings": emb_tensor[idx_tr],
        "mask": mask_tensor[idx_tr],
        "labels": torch.tensor(labels[idx_tr], dtype=torch.float32),
        "sequences": [sequencias[i] for i in idx_tr],
        "max_len": max_len,
        "model_name": MODEL_NAME,
    }
    pacote_teste = {
        "embeddings": emb_tensor[idx_te],
        "mask": mask_tensor[idx_te],
        "labels": torch.tensor(labels[idx_te], dtype=torch.float32),
        "sequences": [sequencias[i] for i in idx_te],
        "max_len": max_len,
        "model_name": MODEL_NAME,
    }

    torch.save(pacote_treino, OUT_TRAIN_PT)
    torch.save(pacote_teste, OUT_TEST_PT)

    mb_tr = OUT_TRAIN_PT.stat().st_size / (1024**2)
    mb_te = OUT_TEST_PT.stat().st_size / (1024**2)

    print(f"\n✅ Concluído em {time.time() - t0:.1f}s (alinhamento 100% validado)!")
    print(f"  -> {OUT_TRAIN_PT.relative_to(ROOT)}: {tuple(pacote_treino['embeddings'].shape)} ({mb_tr:.1f} MB)")
    print(f"  -> {OUT_TEST_PT.relative_to(ROOT)}:  {tuple(pacote_teste['embeddings'].shape)} ({mb_te:.1f} MB)")


if __name__ == "__main__":
    gerar_e_dividir_esm2()
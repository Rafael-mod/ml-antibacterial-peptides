"""
src/features/esm2_total_split.py

Extrai embeddings de toda a base e divide em Treino/Teste 
reproduzindo exatamente o split original.
"""

import pandas as pd
import torch
from transformers import AutoTokenizer, EsmModel
from sklearn.model_selection import train_test_split
from pathlib import Path
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"

CSV_POS = PROC / "dataset_positivo.csv"
CSV_NEG = PROC / "dataset_negativo.csv"

def gerar_e_dividir_esm2():
    # 1. Recria a base total na mesma ordem original
    df_pos = pd.read_csv(CSV_POS)
    df_neg = pd.read_csv(CSV_NEG)
    df_total = pd.concat([df_pos, df_neg], ignore_index=True)
    sequencias = df_total['sequence'].tolist()
    labels = df_total['label'].tolist()

    # 2. Configura o ESM-2
    print("Carregando modelo ESM-2...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    model = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    # 3. Extrai embeddings para toda a base
    embeddings = []
    print(f"Extraindo embeddings para {len(sequencias)} sequências...")
    with torch.no_grad():
        for seq in tqdm(sequencias):
            inputs = tokenizer(seq, return_tensors="pt", padding=True, truncation=True, max_length=1024)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            
            tamanho_real = (inputs['attention_mask'][0] == 1).sum().item() - 2
            if tamanho_real > 0:
                emb_medio = outputs.last_hidden_state[0, 1:tamanho_real+1, :].mean(dim=0)
            else:
                emb_medio = outputs.last_hidden_state[0].mean(dim=0)
            embeddings.append(emb_medio.cpu().numpy())

    df_embeddings = pd.DataFrame(embeddings)

    # 4. Divide os dados com a MESMA SEMENTE do seu script original
    # ATENÇÃO: Altere test_size e random_state para baterem com o seu split original!
    print("\nDividindo dados (train_test_split)...")
    X_train, X_test = train_test_split(
        df_embeddings, 
        test_size=0.20,      # <-- AJUSTE AQUI SE NECESSÁRIO
        random_state=42,     # <-- COLOQUE A MESMA SEMENTE AQUI
        stratify=labels
    )

    # 5. Salva os arquivos finais
    X_train.to_csv(PROC / "X_train_esm2.csv", index=False)
    X_test.to_csv(PROC / "X_test_esm2.csv", index=False)
    
    print(f"\n✅ Concluído!")
    print(f"  -> X_train_esm2: {X_train.shape}")
    print(f"  -> X_test_esm2: {X_test.shape}")

if __name__ == "__main__":
    gerar_e_dividir_esm2()
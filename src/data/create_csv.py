"""
src/data/create_csvs.py

Geração dos Datasets em Formato CSV:
1. Lê 'positivos_finais.fasta' e gera 'dataset_positivo.csv' com a coluna 'group'.
2. Lê 'negativos_finais.fasta' e gera 'dataset_negativo.csv' SEM a coluna 'group'.

Uso:
    python src/data/create_csvs.py
"""

from pathlib import Path
import pandas as pd
from Bio import SeqIO

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"

POS_FASTA = PROC / "positivos_finais.fasta"
NEG_FASTA = PROC / "negativos_finais.fasta"

CSV_POS = PROC / "dataset_positivo.csv"
CSV_NEG = PROC / "dataset_negativo.csv"


def exportar_csvs() -> None:
    print("=== Gerando Arquivos CSV ===")

    if not POS_FASTA.exists() or not NEG_FASTA.exists():
        raise FileNotFoundError(
            "Arquivos FASTA finais não encontrados em data/processed/. "
            "Rode o script 'preprocess.py' primeiro."
        )

    # 1. Dataset Positivo
    dados_pos = []
    for r in SeqIO.parse(str(POS_FASTA), "fasta"):
        if "real_ambos" in r.id:
            grupo = "gram_ambos"
        elif "only_gram_neg" in r.id:
            grupo = "gram_neg"
        else:
            grupo = "gram_pos"

        dados_pos.append({
            "id": r.id,
            "sequence": str(r.seq),
            "length": len(r.seq),
            "group": grupo,
            "label": 1
        })

    df_pos = pd.DataFrame(dados_pos)
    df_pos.to_csv(CSV_POS, index=False)
    print(f"  -> {CSV_POS.name} criado ({len(df_pos)} linhas)")
    print(df_pos["group"].value_counts().to_string())

    # 2. Dataset Negativo (SEM a coluna 'group')
    dados_neg = []
    for r in SeqIO.parse(str(NEG_FASTA), "fasta"):
        dados_neg.append({
            "id": r.id,
            "sequence": str(r.seq),
            "length": len(r.seq),
            "label": 0
        })

    df_neg = pd.DataFrame(dados_neg)
    df_neg.to_csv(CSV_NEG, index=False)
    print(f"\n  -> {CSV_NEG.name} criado ({len(df_neg)} linhas)")


def main() -> None:
    exportar_csvs()
    print("\n✅ Datasets CSV criados com sucesso!")

if __name__ == "__main__":
    main()
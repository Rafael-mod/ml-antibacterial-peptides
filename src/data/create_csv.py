"""
src/data/create_csvs.py

Geração dos Datasets em Formato CSV com Mapeamento de Clusters (40%):
1. Lê 'clusters_40.fasta.clstr' para atribuir a coluna 'cluster_40' a cada peptídeo.
2. Lê 'positivos_finais.fasta' e gera 'dataset_positivo.csv' com 'group' e 'cluster_40'.
3. Lê 'negativos_finais.fasta' e gera 'dataset_negativo.csv' com 'cluster_40' (SEM 'group').

Uso:
    python src/data/create_csvs.py
"""

import re
from pathlib import Path

import pandas as pd
from Bio import SeqIO

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"

POS_FASTA = PROC / "positivos_finais.fasta"
NEG_FASTA = PROC / "negativos_finais.fasta"
CLSTR_40 = PROC / "clusters_40.fasta.clstr"

CSV_POS = PROC / "dataset_positivo.csv"
CSV_NEG = PROC / "dataset_negativo.csv"


def ler_mapa_clusters(clstr_path: Path) -> dict[str, int]:
    """Lê o arquivo .clstr do CD-HIT e retorna {seq_id: cluster_id}."""
    mapa: dict[str, int] = {}
    cluster_atual: int | None = None

    with clstr_path.open("r", encoding="utf-8") as f:
        for linha in f:
            linha = linha.strip()
            if not linha:
                continue
            if linha.startswith(">Cluster"):
                cluster_atual = int(linha.split()[1])
            else:
                match = re.search(r">(.+?)\.\.\.", linha)
                if match and cluster_atual is not None:
                    seq_id = match.group(1)
                    mapa[seq_id] = cluster_atual

    return mapa


def obter_cluster_estrito(seq_id: str, mapa_clusters: dict[str, int]) -> int:
    if seq_id not in mapa_clusters:
        raise KeyError(
            f"Sequência '{seq_id}' não encontrada em {CLSTR_40.name}. "
            "Rode 'preprocess_fasta.py' novamente."
        )
    return mapa_clusters[seq_id]


def exportar_csvs() -> None:
    print("=== Gerando Arquivos CSV ===")

    for arq in (POS_FASTA, NEG_FASTA, CLSTR_40):
        if not arq.exists():
            raise FileNotFoundError(
                f"Arquivo '{arq.name}' não encontrado em data/processed/. "
                "Rode o script 'preprocess_fasta.py' primeiro."
            )

    mapa_clusters = ler_mapa_clusters(CLSTR_40)

    # 1. Dataset Positivo
    dados_pos = []
    for r in SeqIO.parse(str(POS_FASTA), "fasta"):
        if "real_ambos" in r.id:
            grupo = "gram_ambos"
        elif "only_gram_neg" in r.id:
            grupo = "gram_neg"
        elif "only_gram_pos" in r.id:
            grupo = "gram_pos"
        else:
            raise ValueError(f"ID positivo com prefixo desconhecido: {r.id}")

        dados_pos.append({
            "id": r.id,
            "sequence": str(r.seq),
            "length": len(r.seq),
            "group": grupo,
            "cluster_40": obter_cluster_estrito(r.id, mapa_clusters),
            "label": 1,
        })

    df_pos = pd.DataFrame(dados_pos)
    df_pos.to_csv(CSV_POS, index=False)
    print(f"  -> {CSV_POS.name} criado ({len(df_pos)} linhas | {df_pos['cluster_40'].nunique()} clusters 40%)")
    print(df_pos["group"].value_counts().to_string())

    # 2. Dataset Negativo (SEM a coluna 'group')
    dados_neg = []
    for r in SeqIO.parse(str(NEG_FASTA), "fasta"):
        dados_neg.append({
            "id": r.id,
            "sequence": str(r.seq),
            "length": len(r.seq),
            "cluster_40": obter_cluster_estrito(r.id, mapa_clusters),
            "label": 0,
        })

    df_neg = pd.DataFrame(dados_neg)
    df_neg.to_csv(CSV_NEG, index=False)
    print(f"\n  -> {CSV_NEG.name} criado ({len(df_neg)} linhas | {df_neg['cluster_40'].nunique()} clusters 40%)")

    # Diagnóstico rápido de sobreposição de clusters entre classes
    clusters_mistos = set(df_pos["cluster_40"]).intersection(set(df_neg["cluster_40"]))
    print(f"  -> Clusters 40% compartilhados entre Pos e Neg: {len(clusters_mistos)}")


def main() -> None:
    exportar_csvs()
    print("\n✅ Datasets CSV criados com sucesso!")


if __name__ == "__main__":
    main()
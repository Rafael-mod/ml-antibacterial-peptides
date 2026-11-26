"""
src/data/preprocess_fasta.py

Pipeline de Pré-processamento (FASTA -> CD-HIT -> FASTAs Finais + Clusters 40%):
1. Aplica lógica de conjuntos nas sequências positivas (exclusividade de grupos).
2. Filtra por tamanho (8-50 AA) e aminoácidos canônicos.
3. Roda CD-HIT (80%) por grupo positivo, seguido de CD-HIT global (80%) para
   eliminar redundância cruzada entre grupos, e CD-HIT-2D contra o PGAT-ABPp.
4. Lê os negativos naturais completos e maduros do UniProt (sem recorte artificial).
5. Roda CD-HIT interno (80%) e CD-HIT-2D (80%) contra os positivos e contra o
   benchmark externo PGAT-ABPp (0% vazamento >= 80% em ambas as classes).
6. Aplica Pareamento Bidirecional Estrito por Comprimento (Mutual Length-Matching):
   para cada tamanho L in [8, 50], seleciona min(N_pos(L), N_neg(L)) em ambas as classes,
   garantindo histograma, média e desvio padrão idênticos entre Positivos e Negativos.
7. Salva 'positivos_finais.fasta', 'negativos_finais.fasta' e gera
   'clusters_40.fasta.clstr' (CD-HIT 40%) em data/processed/ para split rigoroso por homologia.

Uso:
    python src/data/preprocess_fasta.py
"""

import random
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

# ==========================================
# Configurações de Diretórios e Parâmetros
# ==========================================
ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PRE = ROOT / "data" / "pre-processed"
PROC = ROOT / "data" / "processed"
EXT_PGAT = ROOT / "data" / "external" / "pgat"

IN_GRAM_NEG = RAW / "only_gram_neg.fasta"
IN_GRAM_POS = RAW / "only_gram_pos.fasta"
IN_AMBOS = RAW / "gram_neg_pos.fasta"
IN_UNIPROT = RAW / "uniprot_negativos_naturais.fasta"

PGAT_POS_FASTA = EXT_PGAT / "pgat_positivos_finais.fasta"
PGAT_NEG_FASTA = EXT_PGAT / "pgat_negativos_finais.fasta"

TAMANHO_MIN = 8
TAMANHO_MAX = 50

# Parâmetros CD-HIT (redução de redundância no pool de treino/teste)
CDHIT_C = 0.8
CDHIT_N = 3
CDHIT_THREADS = 0

# Parâmetros CD-HIT (blindagem contra benchmark externo PGAT-ABPp)
CDHIT_PGAT_C = 0.8
CDHIT_PGAT_N = 3

# Parâmetros CD-HIT (agrupamento a 40% para split/CV sem vazamento de homologia)
CDHIT_SPLIT_C = 0.4
CDHIT_SPLIT_N = 2

SEED = 42
PADRAO_CANONICOS = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$")
PREFIXOS_POSITIVOS = ("only_gram_neg_", "only_gram_pos_", "real_ambos_")


# ==========================================
# Funções Auxiliares
# ==========================================
def ler_fasta_raw(arquivo: Path) -> set[str]:
    if not arquivo.exists():
        print(f"  [!] Arquivo não encontrado: {arquivo}")
        return set()
    return {str(r.seq).upper() for r in SeqIO.parse(str(arquivo), "fasta")}


def filtrar_sequencias(seqs: set[str]) -> list[str]:
    validas = [
        s for s in seqs
        if TAMANHO_MIN <= len(s) <= TAMANHO_MAX and PADRAO_CANONICOS.match(s)
    ]
    return sorted(validas, key=lambda s: (len(s), s))


def salvar_fasta(seqs: list[str], arquivo: Path, prefixo: str) -> None:
    arquivo.parent.mkdir(parents=True, exist_ok=True)
    records = [
        SeqRecord(Seq(s), id=f"{prefixo}_{i}", description="")
        for i, s in enumerate(seqs, 1)
    ]
    SeqIO.write(records, str(arquivo), "fasta")
    print(f"  -> {arquivo.name}: {len(records)} sequências")


def checar_cdhit() -> None:
    for exe in ("cd-hit", "cd-hit-2d"):
        if shutil.which(exe) is None:
            raise FileNotFoundError(f"Erro: '{exe}' não encontrado no PATH.")


def montar_args_cdhit(c: float, n: int) -> list[str]:
    # -l (TAMANHO_MIN - 1) evita que o CD-HIT descarte silenciosamente sequências <= 10 AA
    return [
        "-c", str(c),
        "-n", str(n),
        "-l", str(TAMANHO_MIN - 1),
        "-d", "0",
        "-M", "0",
        "-T", str(CDHIT_THREADS),
    ]


def rodar_cdhit(cmd: list[str]) -> None:
    print("  $ " + " ".join(cmd))
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)


# ==========================================
# [1/4] Pool Limpo de Positivos
# ==========================================
def obter_pool_positivos() -> list[SeqRecord]:
    print("\n=== [1/4] Positivos: Filtros, Exclusividade, CD-HIT Global e Blindagem PGAT ===")

    gram_neg_raw = ler_fasta_raw(IN_GRAM_NEG)
    gram_pos_raw = ler_fasta_raw(IN_GRAM_POS)
    ambos_raw = ler_fasta_raw(IN_AMBOS)

    real_ambos = ambos_raw - gram_neg_raw - gram_pos_raw
    only_neg = gram_neg_raw - gram_pos_raw
    only_pos = gram_pos_raw - gram_neg_raw

    print(f"  Brutos -> Neg: {len(gram_neg_raw)} | Pos: {len(gram_pos_raw)} | Ambos: {len(ambos_raw)}")
    print(f"  Exclusivos -> Neg: {len(only_neg)} | Pos: {len(only_pos)} | Ambos (real): {len(real_ambos)}")

    salvar_fasta(filtrar_sequencias(only_neg), PRE / "only_gram_neg_filtrado.fasta", "gram_neg")
    salvar_fasta(filtrar_sequencias(only_pos), PRE / "only_gram_pos_filtrado.fasta", "gram_pos")
    salvar_fasta(filtrar_sequencias(real_ambos), PRE / "real_ambos_filtrado.fasta", "real_ambos")

    args_cdhit = montar_args_cdhit(CDHIT_C, CDHIT_N)
    pool_por_grupo: list[SeqRecord] = []

    for nome in ["only_gram_neg", "only_gram_pos", "real_ambos"]:
        in_file = PRE / f"{nome}_filtrado.fasta"
        out_file = PRE / f"{nome}_cdhit.fasta"
        rodar_cdhit(["cd-hit", "-i", str(in_file), "-o", str(out_file), *args_cdhit])
        recs = list(SeqIO.parse(str(out_file), "fasta"))
        print(f"  CD-HIT ({nome}): {len(recs)} sequências restantes")
        for i, rec in enumerate(recs, 1):
            rec.id = f"{nome}_{i}"
            rec.description = ""
        pool_por_grupo.extend(recs)

    # CD-HIT global para eliminar redundância cruzada entre os três grupos positivos
    pos_concat = PRE / "positivos_concatenados.fasta"
    pos_global = PRE / "positivos_global_cdhit.fasta"
    SeqIO.write(pool_por_grupo, str(pos_concat), "fasta")
    rodar_cdhit(["cd-hit", "-i", str(pos_concat), "-o", str(pos_global), *args_cdhit])

    arquivo_corrente = pos_global
    n_global = len(list(SeqIO.parse(str(pos_global), "fasta")))
    print(f"  -> Após CD-HIT global (80%, sem redundância cruzada): {n_global} sequências (removidas {len(pool_por_grupo) - n_global})")

    # Blindagem dos positivos contra o benchmark externo PGAT-ABPp
    if PGAT_POS_FASTA.exists() and PGAT_NEG_FASTA.exists():
        args_pgat = montar_args_cdhit(CDHIT_PGAT_C, CDHIT_PGAT_N)
        pos_vs_pgat1 = PRE / "positivos_vs_pgat_pos.fasta"
        pos_vs_pgat2 = PRE / "positivos_vs_pgat_neg.fasta"

        rodar_cdhit([
            "cd-hit-2d", "-i", str(PGAT_POS_FASTA), "-i2", str(arquivo_corrente),
            "-o", str(pos_vs_pgat1), *args_pgat
        ])
        rodar_cdhit([
            "cd-hit-2d", "-i", str(PGAT_NEG_FASTA), "-i2", str(pos_vs_pgat1),
            "-o", str(pos_vs_pgat2), *args_pgat
        ])
        arquivo_corrente = pos_vs_pgat2
        print(f"  -> Blindagem CD-HIT-2D aplicada nos positivos contra PGAT-ABPp (0% vazamento >= {int(CDHIT_PGAT_C * 100)}%).")

    pool_positivos = [
        rec for rec in SeqIO.parse(str(arquivo_corrente), "fasta")
        if rec.id.startswith(PREFIXOS_POSITIVOS)
    ]
    print(f"\n  -> Pool Positivo Limpo e Blindado: {len(pool_positivos)} sequências.")
    return pool_positivos


# ==========================================
# [2/4] Pool Limpo de Negativos Naturais
# ==========================================
def obter_pool_negativos(pool_positivos_path: Path) -> list[str]:
    print("\n=== [2/4] Negativos Naturais: Filtros e Blindagem Tripla CD-HIT-2D ===")

    if not IN_UNIPROT.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {IN_UNIPROT}")

    seqs_pos = {str(r.seq).upper() for r in SeqIO.parse(str(pool_positivos_path), "fasta")}
    neg_raw = ler_fasta_raw(IN_UNIPROT)
    neg_filtradas = filtrar_sequencias(neg_raw - seqs_pos)
    print(f"  -> Brutas únicas: {len(neg_raw)} | Após filtro canônico (8-50 AA): {len(neg_filtradas)}")

    pool_natural_path = PRE / "negativos_naturais_filtrados.fasta"
    salvar_fasta(neg_filtradas, pool_natural_path, "nat_neg")

    args_cdhit = montar_args_cdhit(CDHIT_C, CDHIT_N)
    neg_cdhit = PRE / "negativos_naturais_cdhit.fasta"
    rodar_cdhit(["cd-hit", "-i", str(pool_natural_path), "-o", str(neg_cdhit), *args_cdhit])
    print(f"  -> Após CD-HIT interno (80%): {len(list(SeqIO.parse(str(neg_cdhit), 'fasta')))} clusters únicos")

    neg_vs_pos = PRE / "negativos_naturais_vs_pos.fasta"
    rodar_cdhit([
        "cd-hit-2d", "-i", str(pool_positivos_path), "-i2", str(neg_cdhit),
        "-o", str(neg_vs_pos), *args_cdhit
    ])
    arquivo_corrente = neg_vs_pos

    if PGAT_POS_FASTA.exists() and PGAT_NEG_FASTA.exists():
        args_pgat = montar_args_cdhit(CDHIT_PGAT_C, CDHIT_PGAT_N)
        neg_vs_pgat1 = PRE / "negativos_naturais_vs_pgat_pos.fasta"
        neg_vs_pgat2 = PRE / "negativos_naturais_vs_pgat_neg.fasta"

        rodar_cdhit([
            "cd-hit-2d", "-i", str(PGAT_POS_FASTA), "-i2", str(arquivo_corrente),
            "-o", str(neg_vs_pgat1), *args_pgat
        ])
        rodar_cdhit([
            "cd-hit-2d", "-i", str(PGAT_NEG_FASTA), "-i2", str(neg_vs_pgat1),
            "-o", str(neg_vs_pgat2), *args_pgat
        ])
        arquivo_corrente = neg_vs_pgat2
        print(f"  -> Blindagem CD-HIT-2D aplicada nos negativos contra PGAT-ABPp (0% vazamento >= {int(CDHIT_PGAT_C * 100)}%).")

    negativos_limpos = [
        str(r.seq) for r in SeqIO.parse(str(arquivo_corrente), "fasta")
        if r.id.startswith("nat_neg_")
    ]
    print(f"  -> Pool Negativo Limpo Disponível: {len(negativos_limpos)} sequências.")
    return negativos_limpos


# ==========================================
# [3/4] Pareamento Bidirecional Estrito por Tamanho
# ==========================================
def parear_por_tamanho_estrito(
    pool_positivos: list[SeqRecord],
    pool_negativos: list[str],
) -> tuple[list[SeqRecord], list[SeqRecord]]:
    print("\n=== [3/4] Pareamento Bidirecional Estrito (Mutual Length-Matching) ===")

    rng = random.Random(SEED)

    pos_por_tam: dict[int, list[SeqRecord]] = {}
    for rec in pool_positivos:
        pos_por_tam.setdefault(len(rec.seq), []).append(rec)

    neg_por_tam: dict[int, list[str]] = {}
    for seq in pool_negativos:
        neg_por_tam.setdefault(len(seq), []).append(seq)

    for tam in pos_por_tam:
        rng.shuffle(pos_por_tam[tam])
    for tam in neg_por_tam:
        rng.shuffle(neg_por_tam[tam])

    positivos_pareados: list[SeqRecord] = []
    negativos_pareados: list[SeqRecord] = []

    for tam in range(TAMANHO_MIN, TAMANHO_MAX + 1):
        lista_p = pos_por_tam.get(tam, [])
        lista_n = neg_por_tam.get(tam, [])
        n_pareado = min(len(lista_p), len(lista_n))

        if n_pareado == 0:
            continue

        positivos_pareados.extend(lista_p[:n_pareado])
        for i, seq_n in enumerate(lista_n[:n_pareado], 1):
            negativos_pareados.append(
                SeqRecord(Seq(seq_n), id=f"negativo_{tam}_{i}", description="")
            )

    tams_pos = [len(r.seq) for r in positivos_pareados]
    tams_neg = [len(r.seq) for r in negativos_pareados]
    faixa_curta = sum(1 for t in tams_pos if 10 <= t <= 30)

    print(f"  -> Positivos Finais Pareados: {len(positivos_pareados)} sequências")
    print(f"  -> Negativos Finais Pareados: {len(negativos_pareados)} sequências")
    print(f"  -> Sequências na faixa crítica (10-30 AA): {faixa_curta} por classe")
    print(
        f"  -> Controle de Tamanho: Positivos = {np.mean(tams_pos):.2f} ± {np.std(tams_pos):.2f} AA | "
        f"Negativos = {np.mean(tams_neg):.2f} ± {np.std(tams_neg):.2f} AA"
    )
    return positivos_pareados, negativos_pareados


# ==========================================
# [4/4] Clusterização a 40% para Split Rigoroso
# ==========================================
def gerar_clusters_40(
    pos_finais: list[SeqRecord],
    neg_finais: list[SeqRecord],
) -> Path:
    print("\n=== [4/4] Gerando Clusters a 40% de Identidade (Sem Descarte de Dados) ===")

    completo_path = PRE / "dataset_final_para_cluster40.fasta"
    out_cluster_fasta = PROC / "clusters_40.fasta"
    SeqIO.write(pos_finais + neg_finais, str(completo_path), "fasta")

    args_split = montar_args_cdhit(CDHIT_SPLIT_C, CDHIT_SPLIT_N)
    rodar_cdhit(["cd-hit", "-i", str(completo_path), "-o", str(out_cluster_fasta), *args_split])

    n_clusters = len(list(SeqIO.parse(str(out_cluster_fasta), "fasta")))
    clstr_path = Path(f"{out_cluster_fasta}.clstr")
    print(f"  -> Total de sequências mapeadas: {len(pos_finais) + len(neg_finais)} em {n_clusters} clusters (40%)")
    print(f"  -> Mapa salvo em: {clstr_path.relative_to(ROOT)}")
    return clstr_path


def main() -> None:
    checar_cdhit()
    PRE.mkdir(parents=True, exist_ok=True)
    PROC.mkdir(parents=True, exist_ok=True)

    pool_positivos = obter_pool_positivos()
    pool_pos_temp = PRE / "pool_positivos_temp.fasta"
    SeqIO.write(pool_positivos, str(pool_pos_temp), "fasta")

    pool_negativos = obter_pool_negativos(pool_pos_temp)
    pos_finais, neg_finais = parear_por_tamanho_estrito(pool_positivos, pool_negativos)

    SeqIO.write(pos_finais, str(PROC / "positivos_finais.fasta"), "fasta")
    SeqIO.write(neg_finais, str(PROC / "negativos_finais.fasta"), "fasta")

    gerar_clusters_40(pos_finais, neg_finais)
    print(f"\n✅ FASTAs finais balanceados e clusters 40% salvos em: {PROC}")


if __name__ == "__main__":
    main()
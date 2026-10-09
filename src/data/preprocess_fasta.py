"""
src/data/preprocess.py

Pipeline de Pré-processamento (FASTA -> CD-HIT -> FASTAs Finais):
1. Aplica lógica de conjuntos nas sequências positivas (exclusividade de grupos).
2. Filtra por tamanho (8-50 AA) e aminoácidos canônicos.
3. Roda CD-HIT individualmente em cada grupo positivo e balanceia pelo menor grupo.
4. Gera o pool de negativos do UniProt com base na distribuição de tamanhos dos positivos.
5. Roda CD-HIT e CD-HIT-2D para filtrar falsos negativos.
6. Aplica downsampling para igualar a distribuição de tamanhos.
7. Salva 'positivos_finais.fasta' e 'negativos_finais.fasta' em data/processed/.

Uso:
    python src/data/preprocess.py
"""

import random
import re
import shutil
import subprocess
from collections import Counter
from pathlib import Path

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

IN_GRAM_NEG = RAW / "only_gram_neg.fasta"
IN_GRAM_POS = RAW / "only_gram_pos.fasta"
IN_AMBOS = RAW / "gram_neg_pos.fasta"
IN_UNIPROT = RAW / "negativo_uniprot.fasta"

TAMANHO_MIN = 8
TAMANHO_MAX = 50

# Parâmetros CD-HIT
CDHIT_C = 0.8
CDHIT_N = 3
CDHIT_THREADS = 0

SEED = 42
FATOR_INCHACO = 5

AMINOACIDOS_CANONICOS = set("ACDEFGHIKLMNPQRSTVWY")
PADRAO_CANONICOS = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$")


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

def rodar_cdhit(cmd: list[str]) -> None:
    print("  $ " + " ".join(cmd))
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)


# ==========================================
# Processamento de Positivos
# ==========================================
def processar_positivos() -> Path:
    print("\n=== [1/3] Positivos: Filtros, Exclusividade e CD-HIT ===")
    
    gram_neg_raw = ler_fasta_raw(IN_GRAM_NEG)
    gram_pos_raw = ler_fasta_raw(IN_GRAM_POS)
    ambos_raw = ler_fasta_raw(IN_AMBOS)

    real_ambos = ambos_raw - gram_neg_raw - gram_pos_raw
    only_neg = gram_neg_raw - gram_pos_raw 
    only_pos = gram_pos_raw - gram_neg_raw 

    print(f"  Brutos -> Neg: {len(gram_neg_raw)} | Pos: {len(gram_pos_raw)} | Ambos: {len(ambos_raw)}")
    print(f"  Exclusivos -> Neg: {len(only_neg)} | Pos: {len(only_pos)} | Ambos (real): {len(real_ambos)}")

    f_neg = filtrar_sequencias(only_neg)
    f_pos = filtrar_sequencias(only_pos)
    f_amb = filtrar_sequencias(real_ambos)

    salvar_fasta(f_neg, PRE / "only_gram_neg_filtrado.fasta", "gram_neg")
    salvar_fasta(f_pos, PRE / "only_gram_pos_filtrado.fasta", "gram_pos")
    salvar_fasta(f_amb, PRE / "real_ambos_filtrado.fasta", "real_ambos")

    arquivos_cdhit = {}
    args_cdhit = ["-c", str(CDHIT_C), "-n", str(CDHIT_N), "-d", "0", "-M", "0", "-T", str(CDHIT_THREADS)]

    for nome in ["only_gram_neg", "only_gram_pos", "real_ambos"]:
        in_file = PRE / f"{nome}_filtrado.fasta"
        out_file = PRE / f"{nome}_cdhit.fasta"
        rodar_cdhit(["cd-hit", "-i", str(in_file), "-o", str(out_file), *args_cdhit])
        arquivos_cdhit[nome] = list(SeqIO.parse(str(out_file), "fasta"))
        print(f"  CD-HIT ({nome}): {len(arquivos_cdhit[nome])} sequências restantes")

    menor_tamanho = min(len(seqs) for seqs in arquivos_cdhit.values())
    print(f"\n  [!] Tamanho do menor grupo: {menor_tamanho} sequências/grupo")

    rng = random.Random(SEED)
    positivos_balanceados = []
    
    for nome, records in arquivos_cdhit.items():
        amostra = rng.sample(records, menor_tamanho)
        for i, rec in enumerate(amostra, 1):
            rec.id = f"{nome}_{i}"
            rec.description = ""
        positivos_balanceados.extend(amostra)

    PROC.mkdir(parents=True, exist_ok=True)
    pos_finais_path = PROC / "positivos_finais.fasta"
    SeqIO.write(positivos_balanceados, str(pos_finais_path), "fasta")
    print(f"  -> Positivos Finais Salvos: {len(positivos_balanceados)} sequências.")
    
    return pos_finais_path


# ==========================================
# Processamento de Negativos
# ==========================================
def processar_negativos(pos_finais_path: Path) -> Path:
    print("\n=== [2/3] Negativos: Distribuição, Pool e CD-HIT-2D ===")
    
    tamanhos = [len(r.seq) for r in SeqIO.parse(str(pos_finais_path), "fasta")]
    dist_alvo = dict(Counter(tamanhos))

    proteinas = list(SeqIO.parse(str(IN_UNIPROT), "fasta"))
    rng = random.Random(SEED)
    registros_pool = []
    contador = 1

    for tamanho, meta in dist_alvo.items():
        meta_inchada = meta * FATOR_INCHACO
        validas = [p for p in proteinas if len(p.seq) >= tamanho]
        
        if not validas:
            continue

        gerados = 0
        tentativas = 0
        while gerados < meta_inchada and tentativas < meta_inchada * 100:
            tentativas += 1
            prot = rng.choice(validas)
            inicio = rng.randint(0, len(prot.seq) - tamanho)
            frag = str(prot.seq[inicio:inicio + tamanho]).upper()

            if set(frag) <= AMINOACIDOS_CANONICOS:
                origem = prot.id.split("|")[1] if "|" in prot.id else prot.id
                registros_pool.append(SeqRecord(
                    Seq(frag), id=f"pool_{contador}_len_{tamanho}_from_{origem}", description=""
                ))
                gerados += 1
                contador += 1
                tentativas = 0

    pool_path = PRE / "pool_negativos_inchado.fasta"
    SeqIO.write(registros_pool, str(pool_path), "fasta")

    args_cdhit = ["-c", str(CDHIT_C), "-n", str(CDHIT_N), "-d", "0", "-M", "0", "-T", str(CDHIT_THREADS)]
    neg_cdhit = PRE / "negativos_cdhit.fasta"
    rodar_cdhit(["cd-hit", "-i", str(pool_path), "-o", str(neg_cdhit), *args_cdhit])
    
    neg_2d_raw = PRE / "negativos_2d_temp.fasta"
    rodar_cdhit(["cd-hit-2d", "-i", str(pos_finais_path), "-i2", str(neg_cdhit), "-o", str(neg_2d_raw), *args_cdhit])

    negativos_filtrados = [r for r in SeqIO.parse(str(neg_2d_raw), "fasta") if r.id.startswith("pool_")]
    por_tamanho = {}
    for r in negativos_filtrados:
        por_tamanho.setdefault(len(r.seq), []).append(r)

    negativos_finais = []
    for tamanho, meta in dist_alvo.items():
        disponiveis = por_tamanho.get(tamanho, [])
        if len(disponiveis) >= meta:
            amostra = rng.sample(disponiveis, meta)
        else:
            print(f"  [!] Aviso: Tamanho {tamanho} possui apenas {len(disponiveis)} itens (Meta: {meta}).")
            amostra = disponiveis
        
        for i, rec in enumerate(amostra, 1):
            rec.id = f"negativo_{tamanho}_{i}"
            rec.description = ""
            negativos_finais.append(rec)

    neg_finais_path = PROC / "negativos_finais.fasta"
    SeqIO.write(negativos_finais, str(neg_finais_path), "fasta")
    print(f"  -> Negativos Finais Salvos: {len(negativos_finais)} sequências.")

    return neg_finais_path


def main() -> None:
    checar_cdhit()
    pos_finais = processar_positivos()
    neg_finais = processar_negativos(pos_finais)
    print("\n=== [3/3] Etapa FASTA Concluída! ===")
    print(f"Arquivos gerados em: {PROC}")

if __name__ == "__main__":
    main()
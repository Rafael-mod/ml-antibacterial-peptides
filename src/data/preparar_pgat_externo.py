"""
src/data/preparar_pgat_externo.py

Preparação e Blindagem do Benchmark Externo (PGAT-ABPp):
1. Lê todos os arquivos .pdb em 'data/independent dataset/abp' e 'nonabp' e extrai a sequência 1D.
2. Aplica os mesmos filtros do pipeline original:
   - Deduplicação interna e remoção de conflitos entre classes.
   - Comprimento entre 8 e 50 aminoácidos.
   - Apenas os 20 aminoácidos canônicos.
3. Roda CD-HIT (-c 0.8) internamente para evitar inflação por clusters redundantes.
4. Roda CD-HIT-2D (-c 0.8) contra 'positivos_finais.fasta' e 'negativos_finais.fasta'
   do seu treino para garantir 0% de vazamento por homologia (>= 80%).
5. Salva os FASTAs finais e os CSVs padronizados em 'data/external/pgat/'.

Uso:
    python src/data/preparar_pgat_externo.py
"""

import re
import shutil
import subprocess
from collections import Counter
from pathlib import Path

import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

# ==========================================
# Configurações e Diretórios
# ==========================================
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
IN_PGAT_DIR = DATA_DIR / "independent dataset"
IN_ABP_DIR = IN_PGAT_DIR / "abp"
IN_NONABP_DIR = IN_PGAT_DIR / "nonabp"

PROC_DIR = DATA_DIR / "processed"
REF_POS_FASTA = PROC_DIR / "positivos_finais.fasta"
REF_NEG_FASTA = PROC_DIR / "negativos_finais.fasta"
REF_TREINO_CSV = PROC_DIR / "dataset_treino.csv"

EXT_DIR = DATA_DIR / "external" / "pgat"
PRE_EXT_DIR = EXT_DIR / "intermediarios"

# Mesmos parâmetros do src/data/preprocess.py
TAMANHO_MIN = 8
TAMANHO_MAX = 50
CDHIT_C = 0.8
CDHIT_N = 3
CDHIT_THREADS = 0

AMINOACIDOS_CANONICOS = set("ACDEFGHIKLMNPQRSTVWY")
PADRAO_CANONICOS = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$")

AA_3_PARA_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


# ==========================================
# Funções Auxiliares
# ==========================================
def checar_dependencias() -> None:
    for exe in ("cd-hit", "cd-hit-2d"):
        if shutil.which(exe) is None:
            raise FileNotFoundError(f"Erro: '{exe}' não encontrado no PATH.")

    for pasta in (IN_ABP_DIR, IN_NONABP_DIR):
        if not pasta.exists():
            raise FileNotFoundError(f"Pasta não encontrada: {pasta}")

    for arq in (REF_POS_FASTA, REF_NEG_FASTA):
        if not arq.exists():
            raise FileNotFoundError(f"Arquivo de referência do treino não encontrado: {arq}")


def rodar_cdhit(cmd: list[str]) -> None:
    print("  $ " + " ".join(cmd))
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)


def extrair_sequencia_pdb(caminho_pdb: Path) -> str:
    """
    Extrai a sequência primária 1D lendo os registros ATOM do arquivo .pdb.
    Usa a chave única (cadeia, número_resíduo, código_inserção) para não duplicar átomos.
    """
    residuos_vistos = set()
    sequencia = []

    with open(caminho_pdb, "r", encoding="utf-8", errors="ignore") as f:
        for linha in f:
            if not linha.startswith("ATOM"):
                continue

            altloc = linha[16]
            if altloc not in (" ", "A", "1"):
                continue

            res_nome = linha[17:20].strip().upper()
            cadeia = linha[21]
            res_seq = linha[22:26].strip()
            icode = linha[26]

            chave_residuo = (cadeia, res_seq, icode)
            if chave_residuo not in residuos_vistos:
                residuos_vistos.add(chave_residuo)
                sequencia.append(AA_3_PARA_1.get(res_nome, "X"))

    return "".join(sequencia)


def ordenar_natural(caminho: Path) -> list:
    """Ordena seq_0.pdb, seq_1.pdb, ..., seq_10.pdb numericamente."""
    partes = re.split(r"(\d+)", caminho.stem)
    return [int(p) if p.isdigit() else p for p in partes]


def ler_pasta_pdbs(pasta: Path) -> list[tuple[str, str]]:
    arquivos = sorted(pasta.glob("*.pdb"), key=ordenar_natural)
    if not arquivos:
        raise FileNotFoundError(f"Nenhum arquivo .pdb encontrado em {pasta}")

    extraidos = []
    for pdb in arquivos:
        seq = extrair_sequencia_pdb(pdb)
        if seq:
            extraidos.append((pdb.stem, seq))
    return extraidos


def filtrar_sequencias(pares_id_seq: list[tuple[str, str]], nome_classe: str) -> list[str]:
    total_bruto = len(pares_id_seq)
    seqs_unicas = {seq for _, seq in pares_id_seq}
    duplicadas = total_bruto - len(seqs_unicas)

    fora_tamanho = [s for s in seqs_unicas if not (TAMANHO_MIN <= len(s) <= TAMANHO_MAX)]
    nao_canonicas = [
        s for s in seqs_unicas
        if (TAMANHO_MIN <= len(s) <= TAMANHO_MAX) and not PADRAO_CANONICOS.match(s)
    ]

    validas = sorted(
        [
            s for s in seqs_unicas
            if TAMANHO_MIN <= len(s) <= TAMANHO_MAX and PADRAO_CANONICOS.match(s)
        ],
        key=lambda s: (len(s), s),
    )

    print(f"\n[{nome_classe}] Auditoria de Filtragem:")
    print(f"  -> Arquivos .pdb lidos:          {total_bruto}")
    print(f"  -> Duplicatas exatas removidas:  {duplicadas}")
    print(f"  -> Fora do tamanho (8-50 AA):    {len(fora_tamanho)}")
    print(f"  -> Resíduos não-canônicos:       {len(nao_canonicas)}")
    print(f"  -> Válidas após filtros:         {len(validas)}")

    return validas


def salvar_fasta(seqs: list[str], arquivo: Path, prefixo: str) -> None:
    arquivo.parent.mkdir(parents=True, exist_ok=True)
    records = [
        SeqRecord(Seq(s), id=f"{prefixo}_{i}", description="")
        for i, s in enumerate(seqs, 1)
    ]
    SeqIO.write(records, str(arquivo), "fasta")


# ==========================================
# Pipeline Principal
# ==========================================
def main() -> None:
    print("=== [1/4] Verificando Pastas e Dependências ===")
    checar_dependencias()
    PRE_EXT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n=== [2/4] Extraindo Resíduos 1D dos Arquivos .pdb ===")
    raw_abp = ler_pasta_pdbs(IN_ABP_DIR)
    raw_nonabp = ler_pasta_pdbs(IN_NONABP_DIR)

    # Remove eventuais sequências ambíguas presentes nas duas pastas ao mesmo tempo
    set_abp = {s for _, s in raw_abp}
    set_nonabp = {s for _, s in raw_nonabp}
    conflitos = set_abp & set_nonabp
    if conflitos:
        print(f"  [!] Removendo {len(conflitos)} sequências presentes em ambas as pastas (abp e nonabp).")
        raw_abp = [(i, s) for i, s in raw_abp if s not in conflitos]
        raw_nonabp = [(i, s) for i, s in raw_nonabp if s not in conflitos]

    seqs_pos_filt = filtrar_sequencias(raw_abp, "PGAT Positivos (abp)")
    seqs_neg_filt = filtrar_sequencias(raw_nonabp, "PGAT Negativos (nonabp)")

    pos_filt_fasta = PRE_EXT_DIR / "pgat_pos_filtrado.fasta"
    neg_filt_fasta = PRE_EXT_DIR / "pgat_neg_filtrado.fasta"
    salvar_fasta(seqs_pos_filt, pos_filt_fasta, "pgat_pos")
    salvar_fasta(seqs_neg_filt, neg_filt_fasta, "pgat_neg")

    print("\n=== [3/4] Redundância Interna (CD-HIT 80%) e Cruzada contra Treino (CD-HIT-2D 80%) ===")
    args_cdhit = ["-c", str(CDHIT_C), "-n", str(CDHIT_N), "-d", "0", "-M", "0", "-T", str(CDHIT_THREADS)]

    # 3.1 CD-HIT interno
    pos_cdhit_int = PRE_EXT_DIR / "pgat_pos_cdhit_interno.fasta"
    neg_cdhit_int = PRE_EXT_DIR / "pgat_neg_cdhit_interno.fasta"
    rodar_cdhit(["cd-hit", "-i", str(pos_filt_fasta), "-o", str(pos_cdhit_int), *args_cdhit])
    rodar_cdhit(["cd-hit", "-i", str(neg_filt_fasta), "-o", str(neg_cdhit_int), *args_cdhit])

    n_pos_int = len(list(SeqIO.parse(str(pos_cdhit_int), "fasta")))
    n_neg_int = len(list(SeqIO.parse(str(neg_cdhit_int), "fasta")))
    print(f"  -> Após CD-HIT interno (80%): Positivos = {n_pos_int} | Negativos = {n_neg_int}")

    # 3.2 CD-HIT-2D contra o dataset de treino/teste do projeto
    pos_sem_vazamento = PRE_EXT_DIR / "pgat_pos_vs_treino.fasta"
    neg_sem_vazamento_1 = PRE_EXT_DIR / "pgat_neg_vs_neg_treino.fasta"
    neg_sem_vazamento_2 = PRE_EXT_DIR / "pgat_neg_vs_pos_treino.fasta"

    # Remove positivos do PGAT com >= 80% de identidade com positivos_finais.fasta
    rodar_cdhit([
        "cd-hit-2d", "-i", str(REF_POS_FASTA), "-i2", str(pos_cdhit_int),
        "-o", str(pos_sem_vazamento), *args_cdhit
    ])

    # Remove negativos do PGAT com >= 80% de identidade com negativos_finais.fasta...
    rodar_cdhit([
        "cd-hit-2d", "-i", str(REF_NEG_FASTA), "-i2", str(neg_cdhit_int),
        "-o", str(neg_sem_vazamento_1), *args_cdhit
    ])
    # ...e também contra positivos_finais.fasta (evita falsos negativos no benchmark)
    rodar_cdhit([
        "cd-hit-2d", "-i", str(REF_POS_FASTA), "-i2", str(neg_sem_vazamento_1),
        "-o", str(neg_sem_vazamento_2), *args_cdhit
    ])

    seqs_pos_finais = sorted(
        [str(r.seq) for r in SeqIO.parse(str(pos_sem_vazamento), "fasta") if r.id.startswith("pgat_pos_")],
        key=lambda s: (len(s), s),
    )
    seqs_neg_finais = sorted(
        [str(r.seq) for r in SeqIO.parse(str(neg_sem_vazamento_2), "fasta") if r.id.startswith("pgat_neg_")],
        key=lambda s: (len(s), s),
    )

    # Prova final de zero sobreposição exata com dataset_treino.csv
    if REF_TREINO_CSV.exists():
        seqs_treino = set(pd.read_csv(REF_TREINO_CSV)["sequence"])
        assert len(set(seqs_pos_finais) & seqs_treino) == 0, "Erro: positivo externo presente no treino!"
        assert len(set(seqs_neg_finais) & seqs_treino) == 0, "Erro: negativo externo presente no treino!"

    print(f"  -> Após CD-HIT-2D contra seu dataset (0% vazamento >= 80%):")
    print(f"     Positivos PGAT independentes: {len(seqs_pos_finais)} (removidos {n_pos_int - len(seqs_pos_finais)} homólogos)")
    print(f"     Negativos PGAT independentes: {len(seqs_neg_finais)} (removidos {n_neg_int - len(seqs_neg_finais)} homólogos)")

    print("\n=== [4/4] Salvando FASTAs e CSVs Padronizados ===")
    out_pos_fasta = EXT_DIR / "pgat_positivos_finais.fasta"
    out_neg_fasta = EXT_DIR / "pgat_negativos_finais.fasta"
    salvar_fasta(seqs_pos_finais, out_pos_fasta, "pgat_pos")
    salvar_fasta(seqs_neg_finais, out_neg_fasta, "pgat_neg")

    df_pos = pd.DataFrame({
        "id": [f"pgat_pos_{i}" for i in range(1, len(seqs_pos_finais) + 1)],
        "sequence": seqs_pos_finais,
        "length": [len(s) for s in seqs_pos_finais],
        "label": 1,
    })
    df_neg = pd.DataFrame({
        "id": [f"pgat_neg_{i}" for i in range(1, len(seqs_neg_finais) + 1)],
        "sequence": seqs_neg_finais,
        "length": [len(s) for s in seqs_neg_finais],
        "label": 0,
    })
    df_unificado = pd.concat([df_pos, df_neg], ignore_index=True)

    df_pos.to_csv(EXT_DIR / "pgat_dataset_positivo.csv", index=False)
    df_neg.to_csv(EXT_DIR / "pgat_dataset_negativo.csv", index=False)
    df_unificado.to_csv(EXT_DIR / "pgat_dataset_unificado.csv", index=False)

    dist_pos = Counter(df_pos["length"])
    dist_neg = Counter(df_neg["length"])
    print(f"\n✅ Concluído! Arquivos salvos em: {EXT_DIR.relative_to(ROOT)}")
    print(f"  -> Total final: {len(df_pos)} positivos | {len(df_neg)} negativos ({len(df_unificado)} total)")
    print(f"  -> Comprimento médio: Positivos = {df_pos['length'].mean():.1f} AA | Negativos = {df_neg['length'].mean():.1f} AA")
    print(f"  -> Faixa de tamanhos: [{min(dist_pos | dist_neg)} .. {max(dist_pos | dist_neg)}] AA")


if __name__ == "__main__":
    main()
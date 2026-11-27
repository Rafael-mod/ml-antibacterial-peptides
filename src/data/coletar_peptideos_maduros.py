"""
src/data/coletar_peptideos_maduros.py

Coleta Peptídeos Maduros Clivados (FT PEPTIDE / FT CHAIN com start > 1)
do UniProt/Swiss-Prot via API REST para composição da classe negativa:
1. Busca entradas revisadas (reviewed:true, fragment:false) com anotações
   de Peptide ou Chain, excluindo qualquer função/keyword antimicrobiana ou tóxica.
2. Extrai apenas regiões com coordenadas exatas (modifier == EXACT),
   clivadas pós-traducionalmente (start > 1, removendo a Metionina iniciadora
   de tradução de proteínas completas), entre 8 e 50 AA e compostas
   exclusivamente pelos 20 aminoácidos canônicos.
3. Salva 'uniprot_peptideos_maduros.fasta' e 'uniprot_negativos_naturais.fasta'
   em data/raw/, prontos para o preprocess_fasta.py.

Uso:
    python src/data/coletar_peptideos_maduros.py
"""

import re
import time
from pathlib import Path

import numpy as np
import requests
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

# ==========================================
# Configurações e Caminhos
# ==========================================
ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"

OUT_MADUROS = RAW / "uniprot_peptideos_maduros.fasta"
OUT_NEGATIVOS = RAW / "uniprot_negativos_naturais.fasta"

TAMANHO_MIN = 8
TAMANHO_MAX = 50
PADRAO_CANONICOS = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$")

# Controle de viés de localização:
# False -> mantém peptídeos secretados não-antimicrobianos (ex.: hormônios peptídicos)
# True  -> exclui todas as proteínas secretadas (KW-0964)
EXCLUIR_SECRETADOS = False

UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"

_FILTRO_SECRETADOS = "NOT (keyword:KW-0964) " if EXCLUIR_SECRETADOS else ""

QUERY_UNIPROT = (
    '(reviewed:true) AND (fragment:false) AND (ft_peptide:* OR ft_chain:*) '
    'NOT (cc_function:antibacterial) NOT (cc_function:antimicrobial) '
    'NOT (keyword:KW-0048) NOT (keyword:KW-0042) NOT (keyword:KW-0078) '
    f'NOT (keyword:KW-0211) NOT (keyword:KW-0800) {_FILTRO_SECRETADOS}'
    'NOT (keyword:KW-0039) NOT (keyword:KW-0272) '
    'NOT "antimicrobial" NOT "antibacterial" NOT "antifungal" '
    'NOT "antiviral" NOT "antiparasitic" NOT "cytolytic" NOT "hemolytic" '
    'NOT "bacteriocin" NOT "defensin" NOT "cathelicidin" '
    'NOT "toxin" NOT "venom"'
)


# ==========================================
# Cliente HTTP com Retry Automático
# ==========================================
def criar_sessao() -> requests.Session:
    sessao = requests.Session()
    estrategia = Retry(
        total=5,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adaptador = HTTPAdapter(max_retries=estrategia)
    sessao.mount("https://", adaptador)
    return sessao


def extrair_proximo_link(header_link: str | None) -> str | None:
    if not header_link:
        return None
    match = re.search(r'<([^>]+)>;\s*rel="next"', header_link)
    return match.group(1) if match else None


# ==========================================
# Coleta e Parsing dos Peptídeos Maduros Clivados
# ==========================================
def baixar_peptideos_maduros() -> dict[str, str]:
    sessao = criar_sessao()
    params = {
        "query": QUERY_UNIPROT,
        "format": "json",
        "fields": "accession,sequence,ft_peptide,ft_chain",
        "size": 500,
    }

    print("=== [1/2] Consultando API REST do UniProt/Swiss-Prot ===")
    print(f"  -> EXCLUIR_SECRETADOS (KW-0964): {EXCLUIR_SECRETADOS}")
    resp = sessao.get(UNIPROT_SEARCH_URL, params=params, timeout=60)
    resp.raise_for_status()

    total_entradas = int(resp.headers.get("x-total-results", 0))
    print(f"  -> Entradas Swiss-Prot candidatas encontradas: {total_entradas}")

    # Mapa: sequencia -> identificador de origem (ex.: P01189_Peptide_138_150)
    peptideos_unicos: dict[str, str] = {}
    entradas_lidas = 0

    pbar = tqdm(total=total_entradas, desc="Processando entradas UniProt", unit="prot")

    while True:
        dados = resp.json()
        resultados = dados.get("results", [])

        for entrada in resultados:
            acc = entrada.get("primaryAccession", "UNK")
            seq_completa = entrada.get("sequence", {}).get("value", "").upper()
            if not seq_completa:
                continue

            for feat in entrada.get("features", []):
                tipo = feat.get("type")
                if tipo not in ("Peptide", "Chain"):
                    continue

                loc = feat.get("location", {})
                start_obj = loc.get("start", {})
                end_obj = loc.get("end", {})

                # Exige coordenadas experimentais/anotadas exatas
                if start_obj.get("modifier") != "EXACT" or end_obj.get("modifier") != "EXACT":
                    continue

                start = start_obj.get("value")
                end = end_obj.get("value")
                if not (isinstance(start, int) and isinstance(end, int)):
                    continue

                tam = end - start + 1
                # start > 1 garante clivagem pós-traducional (evita Metionina iniciadora)
                if TAMANHO_MIN <= tam <= TAMANHO_MAX and 1 < start <= end <= len(seq_completa):
                    subseq = seq_completa[start - 1 : end]
                    if PADRAO_CANONICOS.match(subseq) and subseq not in peptideos_unicos:
                        peptideos_unicos[subseq] = f"{acc}_{tipo}_{start}_{end}"

        entradas_lidas += len(resultados)
        pbar.update(len(resultados))

        prox_url = extrair_proximo_link(resp.headers.get("Link"))
        if not prox_url:
            break

        resp = sessao.get(prox_url, timeout=60)
        resp.raise_for_status()

    pbar.close()
    print(
        f"  -> Entradas varridas: {entradas_lidas} | "
        f"Peptídeos maduros clivados únicos (8-50 AA, start > 1): {len(peptideos_unicos)}"
    )
    return peptideos_unicos


# ==========================================
# Exportação dos Peptídeos Maduros Clivados
# ==========================================
def main() -> None:
    t0 = time.time()
    RAW.mkdir(parents=True, exist_ok=True)

    peptideos_maduros = baixar_peptideos_maduros()
    itens_ordenados = sorted(peptideos_maduros.items(), key=lambda x: (len(x[0]), x[0]))

    print("\n=== [2/2] Salvando Peptídeos Maduros Clivados ===")
    regs_maduros = [
        SeqRecord(Seq(seq), id=f"mature_{i}_{origem}", description="")
        for i, (seq, origem) in enumerate(itens_ordenados, 1)
    ]
    SeqIO.write(regs_maduros, str(OUT_MADUROS), "fasta")
    print(f"  -> Salvo: {OUT_MADUROS.relative_to(ROOT)} ({len(regs_maduros)} sequências)")

    regs_negativos = [
        SeqRecord(Seq(seq), id=f"uniprot_nat_{i}", description=f"mature_{origem}")
        for i, (seq, origem) in enumerate(itens_ordenados, 1)
    ]
    SeqIO.write(regs_negativos, str(OUT_NEGATIVOS), "fasta")
    print(f"  -> Salvo: {OUT_NEGATIVOS.relative_to(ROOT)} ({len(regs_negativos)} sequências)")

    tamanhos = [len(s) for s in peptideos_maduros]
    faixa_curta = sum(1 for t in tamanhos if 10 <= t <= 30)
    com_m = sum(1 for s in peptideos_maduros if s.startswith("M"))
    pct_m = (100 * com_m / len(tamanhos)) if tamanhos else 0.0

    print(
        f"  -> Comprimento médio bruto: {np.mean(tamanhos):.2f} ± {np.std(tamanhos):.2f} AA "
        f"| Na faixa 10-30 AA: {faixa_curta} | Iniciam com M: {com_m} ({pct_m:.1f}%)"
    )
    print(f"\n✅ Coleta concluída em {time.time() - t0:.1f}s!")


if __name__ == "__main__":
    main()
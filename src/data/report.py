"""
src/data/report.py

Gerador de Relatórios e Auditoria de Datasets (EDA)
Este script lê os CSVs finais, realiza auditoria de vazamento de dados, 
compara distribuições e gera gráficos visuais para análise biológica.

Uso:
    python src/data/report.py
"""

import os
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ==========================================
# Configurações de Diretórios
# ==========================================
ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

CSV_POS = PROC / "dataset_positivo.csv"
CSV_NEG = PROC / "dataset_negativo.csv"

# ==========================================
# Análises Numéricas e de Texto
# ==========================================
def auditar_datasets(df_pos: pd.DataFrame, df_neg: pd.DataFrame) -> None:
    print("="*50)
    print("📊 RELATÓRIO DE AUDITORIA E EDA")
    print("="*50)
    
    total_pos = len(df_pos)
    total_neg = len(df_neg)
    
    print("\n1️⃣ TESTE DE QUANTIDADE E TAMANHO TOTAL")
    print(f"  -> Total de Positivos: {total_pos}")
    print(f"  -> Total de Negativos: {total_neg}")
    if total_pos == total_neg:
        print("  ✅ SUCESSO: Datasets perfeitamente balanceados em volume.")
    else:
        print("  ⚠️ AVISO: Classes desbalanceadas em volume.")

    # --- Teste de Vazamento (Leakage) ---
    print("\n2️⃣ TESTE DE VAZAMENTO (SEQUÊNCIAS IDÊNTICAS)")
    set_pos = set(df_pos['sequence'])
    set_neg = set(df_neg['sequence'])
    sobreposicao = set_pos.intersection(set_neg)
    
    if not sobreposicao:
        print("  ✅ SUCESSO: Zero vazamento! Nenhuma sequência negativa vaza para os positivos.")
    else:
        print(f"  ❌ FALHA: {len(sobreposicao)} sequências estão presentes em AMBOS os datasets!")

    # --- Teste de Pareamento ---
    print("\n3️⃣ TESTE DE PAREAMENTO DE TAMANHOS")
    dist_pos = df_pos['length'].value_counts().sort_index()
    dist_neg = df_neg['length'].value_counts().sort_index()
    
    df_comp = pd.DataFrame({'Positivos': dist_pos, 'Negativos': dist_neg}).fillna(0).astype(int)
    df_comp['Batem'] = df_comp['Positivos'] == df_comp['Negativos']
    
    if df_comp['Batem'].all():
        print("  ✅ SUCESSO: As distribuições de tamanho batem PERFEITAMENTE!")
    else:
        print("  ❌ FALHA: Diferenças encontradas na distribuição de tamanhos.")
        print(df_comp[~df_comp['Batem']])
        
    # --- Distribuição de Grupos ---
    print("\n4️⃣ COMPOSIÇÃO DE GRUPOS (POSITIVOS)")
    print(df_pos['group'].value_counts().to_string())
    print("="*50)

# ==========================================
# Análises Visuais (Gráficos)
# ==========================================
def calcular_composicao_aa(df: pd.DataFrame) -> pd.Series:
    """Calcula a frequência percentual de cada aminoácido num dataset."""
    todas_seqs = "".join(df["sequence"].tolist())
    contagem = Counter(todas_seqs)
    total_aa = sum(contagem.values())
    return pd.Series({aa: (qtd / total_aa) * 100 for aa, qtd in contagem.items()}).sort_index()

def gerar_graficos(df_pos: pd.DataFrame, df_neg: pd.DataFrame) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Gráfico de Distribuição de Tamanhos
    df_pos['Classe'] = 'Positivo'
    df_neg['Classe'] = 'Negativo'
    df_all = pd.concat([df_pos, df_neg])
    
    sns.histplot(
        data=df_all, x="length", hue="Classe", 
        multiple="dodge", bins=range(8, 52, 2), ax=axes[0], palette=["#2ecc71", "#e74c3c"]
    )
    axes[0].set_title("Distribuição de Tamanho das Sequências", fontsize=14)
    axes[0].set_xlabel("Tamanho da Sequência (AA)")
    axes[0].set_ylabel("Frequência")

    # 2. Gráfico de Grupos Positivos (Donut Chart)
    grupos = df_pos['group'].value_counts()
    axes[1].pie(grupos, labels=grupos.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("pastel"))
    axes[1].add_artist(plt.Circle((0,0), 0.70, fc='white')) # Transforma a Pizza em Donut
    axes[1].set_title("Proporção dos Subgrupos (Positivos)", fontsize=14)

    # 3. Composição de Aminoácidos (Comparação)
    aa_pos = calcular_composicao_aa(df_pos)
    aa_neg = calcular_composicao_aa(df_neg)
    
    df_aa = pd.DataFrame({'Positivos': aa_pos, 'Negativos': aa_neg}).fillna(0)
    df_aa.plot(kind='bar', ax=axes[2], color=["#2ecc71", "#e74c3c"], width=0.8)
    
    axes[2].set_title("Composição Global de Aminoácidos (%)", fontsize=14)
    axes[2].set_xlabel("Aminoácido")
    axes[2].set_ylabel("Frequência (%)")
    
    plt.tight_layout()
    
    caminho_grafico = REPORTS / "analise_exploratoria.png"
    plt.savefig(caminho_grafico, dpi=300)
    print(f"\n📈 Gráficos gerados com sucesso e salvos em:\n -> {caminho_grafico}")

# ==========================================
# Execução Principal
# ==========================================
def main():
    if not CSV_POS.exists() or not CSV_NEG.exists():
        print(f"Erro: Arquivos CSV não encontrados em {PROC}")
        return
        
    df_pos = pd.read_csv(CSV_POS)
    df_neg = pd.read_csv(CSV_NEG)
    
    auditar_datasets(df_pos, df_neg)
    gerar_graficos(df_pos, df_neg)

if __name__ == "__main__":
    main()
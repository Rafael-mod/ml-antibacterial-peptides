"""
src/voting/analise_aprofundada_pgat.py

Análise Aprofundada do Benchmark Externo PGAT-ABPp (5.149 peptídeos):
1. Correção Bayesiana de Prevalência (Prior Shift):
   Ajusta matematicamente o limiar OOF (0.605 em 50% de prevalência) para a
   prevalência real do PGAT (18,5% -> tau ≈ 0.871) sem olhar os rótulos individuais.
2. Avaliação Balanceada 1:1 (100 Sorteios Estratificados):
   Avalia Base A, Base B e Meta-Modelo em 100 subamostras balanceadas
   (952 Pos vs. 952 Neg) para comparação direta com o teste cego interno.
3. Inspeção Biofísica dos Top 10 Falsos Positivos:
   Lista os 10 peptídeos Non-ABP com maior score no Meta-Modelo e calcula
   suas propriedades físico-químicas (carga líquida, GRAVY, momento hidrofóbico uH,
   comprimento e fração catiônica) para diagnóstico biológico.

Uso:
    python src/voting/analise_aprofundada_pgat.py
"""

from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from scipy.special import expit, logit
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    roc_auc_score,
)

sns.set_theme(style="whitegrid", context="notebook")

# ==========================================
# Caminhos e Constantes
# ==========================================
ROOT = Path(__file__).resolve().parents[2]
EXT_DIR = ROOT / "data" / "external" / "pgat"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
PASTA_TAB = REPORTS_DIR / "tabelas"
PASTA_FIG = REPORTS_DIR / "figuras_modelagem"

PATH_CSV_PGAT = EXT_DIR / "pgat_dataset_unificado.csv"
PATH_X_FQ = EXT_DIR / "X_pgat_fq.csv"
PATH_X_ESM = EXT_DIR / "X_pgat_esm2.csv"
PATH_Y = EXT_DIR / "y_pgat.csv"

# Escala de hidrofobicidade normalizada de Eisenberg (1984)
EISENBERG = {
    "A": 0.62, "R": -2.53, "N": -0.78, "D": -0.90, "C": 0.29,
    "Q": -0.85, "E": -0.74, "G": 0.48, "H": -0.40, "I": 1.38,
    "L": 1.06, "K": -1.50, "M": 0.64, "F": 1.19, "P": 0.12,
    "S": -0.18, "T": -0.05, "W": 0.81, "Y": 0.26, "V": 1.08,
}


# ==========================================
# Funções Auxiliares
# ==========================================
def probs_para_logodds(probs: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return logit(np.clip(probs, eps, 1.0 - eps))


def ajustar_limiar_bayes(limiar_treino: float, prev_treino: float, prev_externa: float) -> float:
    """
    Corrige o limiar de decisão pela mudança de prevalência (Prior Shift)
    usando a razão de chances a priori do Teorema de Bayes.
    """
    odds_treino = prev_treino / (1.0 - prev_treino)
    odds_externa = prev_externa / (1.0 - prev_externa)
    logit_ajustado = logit(limiar_treino) - np.log(odds_externa / odds_treino)
    return float(expit(logit_ajustado))


def calcular_momento_hidrofobico(seq: str, angulo_graus: float = 100.0) -> float:
    """Calcula o momento hidrofóbico médio (uH) de Eisenberg para hélice alfa (100°)."""
    n = len(seq)
    if n == 0:
        return 0.0
    delta = np.deg2rad(angulo_graus)
    idx = np.arange(1, n + 1)
    h = np.array([EISENBERG.get(aa, 0.0) for aa in seq])
    soma_sen = np.sum(h * np.sin(idx * delta))
    soma_cos = np.sum(h * np.cos(idx * delta))
    return float(np.sqrt(soma_sen**2 + soma_cos**2) / n)


def caracterizar_peptideo(seq: str) -> dict:
    """Extrai propriedades biofísicas interpretáveis de uma sequência peptídica."""
    pa = ProteinAnalysis(seq)
    n = len(seq)
    carga_ph74 = pa.charge_at_pH(7.4)
    gravy = pa.gravy()
    uh_alfa = calcular_momento_hidrofobico(seq, angulo_graus=100.0)
    pct_cationicos = 100.0 * sum(aa in ("K", "R") for aa in seq) / n
    pct_hidrofobicos = 100.0 * sum(aa in ("A", "V", "I", "L", "M", "F", "Y", "W") for aa in seq) / n
    return {
        "Comprimento": n,
        "Carga_pH7.4": round(carga_ph74, 2),
        "GRAVY": round(gravy, 3),
        "Momento_Hidro_uH": round(uh_alfa, 3),
        "Pct_Cationicos_KR": round(pct_cationicos, 1),
        "Pct_Hidrofobicos": round(pct_hidrofobicos, 1),
    }


def resumir_metricas(y_true: np.ndarray, probs: np.ndarray, limiar: float) -> dict:
    y_pred = (probs >= limiar).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "MCC": matthews_corrcoef(y_true, y_pred),
        "AUROC": roc_auc_score(y_true, probs),
        "AUCPR": average_precision_score(y_true, probs),
        "Sensibilidade": tp / (tp + fn),
        "Especificidade": tn / (tn + fp),
        "Precisao": precision_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "TP": int(tp),
        "FN": int(fn),
        "TN": int(tn),
        "FP": int(fp),
    }


# ==========================================
# 1. Análise de Prior Shift (Corte Bayesiano)
# ==========================================
def avaliar_prior_shift(y_pgat: np.ndarray, prob_meta: np.ndarray, limiar_oof: float) -> pd.DataFrame:
    prev_externa = float(y_pgat.mean())
    limiar_bayes = ajustar_limiar_bayes(limiar_oof, prev_treino=0.50, prev_externa=prev_externa)

    cenarios = [
        ("Meta-Modelo (Corte Padrão 0.50)", 0.50),
        (f"Meta-Modelo (Corte OOF Treino {limiar_oof:.3f})", limiar_oof),
        (f"Meta-Modelo (Corte Bayesiano {limiar_bayes:.3f})", limiar_bayes),
    ]

    linhas = []
    for nome, tau in cenarios:
        m = resumir_metricas(y_pgat, prob_meta, tau)
        linhas.append({
            "Cenario": nome,
            "Limiar": round(tau, 3),
            "MCC": round(m["MCC"], 4),
            "Sensibilidade": round(m["Sensibilidade"], 4),
            "Especificidade": round(m["Especificidade"], 4),
            "Precisao": round(m["Precisao"], 4),
            "F1": round(m["F1"], 4),
            "TP": m["TP"],
            "FN": m["FN"],
            "TN": m["TN"],
            "FP": m["FP"],
        })

    df_shift = pd.DataFrame(linhas)
    print("\n=== [1/3] IMPACTO DO PRIOR SHIFT (PREVALÊNCIA 50% -> 18,5%) ===")
    print(f"  -> Prevalência real no PGAT-ABPp: {prev_externa:.4f} ({int(y_pgat.sum())} Pos / {int((y_pgat == 0).sum())} Neg)")
    print(f"  -> Limiar teórico corrigido via Bayes: {limiar_bayes:.3f}")
    print(df_shift.to_string(index=False))
    return df_shift


# ==========================================
# 2. Subamostragem Balanceada 1:1 (100 Runs)
# ==========================================
def avaliar_subamostragem_balanceada(
    y_pgat: np.ndarray,
    prob_fq: np.ndarray,
    prob_esm: np.ndarray,
    prob_meta: np.ndarray,
    limiar_oof: float,
    n_runs: int = 100,
) -> pd.DataFrame:
    idx_pos = np.where(y_pgat == 1)[0]
    idx_neg = np.where(y_pgat == 0)[0]
    n_pos = len(idx_pos)

    modelos = {
        "Base A (FQ LightGBM Calibrado)": (prob_fq, 0.50),
        "Base B (ESM-2 Linear L2)": (prob_esm, 0.50),
        "Meta-Modelo (Corte 0.50)": (prob_meta, 0.50),
        f"Meta-Modelo (Corte OOF {limiar_oof:.2f})": (prob_meta, limiar_oof),
    }

    historico = {nome: [] for nome in modelos}
    rng = np.random.default_rng(42)

    for _ in range(n_runs):
        neg_sorteados = rng.choice(idx_neg, size=n_pos, replace=False)
        idx_bal = np.concatenate([idx_pos, neg_sorteados])
        y_bal = y_pgat[idx_bal]

        for nome, (probs, tau) in modelos.items():
            historico[nome].append(resumir_metricas(y_bal, probs[idx_bal], tau))

    resumo = []
    for nome, lista_m in historico.items():
        df_m = pd.DataFrame(lista_m)
        resumo.append({
            "Modelo": nome,
            "MCC": f"{df_m['MCC'].mean():.3f} ± {df_m['MCC'].std():.3f}",
            "AUROC": f"{df_m['AUROC'].mean():.3f} ± {df_m['AUROC'].std():.3f}",
            "AUCPR": f"{df_m['AUCPR'].mean():.3f} ± {df_m['AUCPR'].std():.3f}",
            "Sensibilidade": f"{df_m['Sensibilidade'].mean():.3f} ± {df_m['Sensibilidade'].std():.3f}",
            "Especificidade": f"{df_m['Especificidade'].mean():.3f} ± {df_m['Especificidade'].std():.3f}",
            "F1": f"{df_m['F1'].mean():.3f} ± {df_m['F1'].std():.3f}",
        })

    df_bal = pd.DataFrame(resumo)
    print(f"\n=== [2/3] AVALIAÇÃO SOB PREVALÊNCIA BALANCEADA 1:1 ({n_runs} SORTEIOS: {n_pos} POS vs. {n_pos} NEG) ===")
    print(df_bal.to_string(index=False))
    return df_bal


# ==========================================
# 3. Inspeção Biofísica dos Top 10 Falsos Positivos
# ==========================================
def inspecionar_top_falsos_positivos(
    df_pgat: pd.DataFrame,
    prob_fq: np.ndarray,
    prob_esm: np.ndarray,
    prob_meta: np.ndarray,
    top_k: int = 10,
) -> pd.DataFrame:
    df_analise = df_pgat.copy()
    df_analise["Prob_FQ"] = prob_fq
    df_analise["Prob_ESM2"] = prob_esm
    df_analise["Prob_Meta"] = prob_meta

    # Filtra apenas os negativos reais (label == 0) e ordena pela maior probabilidade do Meta-Modelo
    df_fp = (
        df_analise[df_analise["label"] == 0]
        .sort_values("Prob_Meta", ascending=False)
        .head(top_k)
        .reset_index(drop=True)
    )

    props = pd.DataFrame([caracterizar_peptideo(s) for s in df_fp["sequence"]])
    df_top_fp = pd.concat(
        [
            df_fp[["id", "sequence", "Prob_FQ", "Prob_ESM2", "Prob_Meta"]].round(4),
            props,
        ],
        axis=1,
    )

    # Perfil biofísico médio de referência dos Positivos vs. Negativos do PGAT
    df_neg_all = df_analise[df_analise["label"] == 0]
    df_pos_all = df_analise[df_analise["label"] == 1]

    props_neg_all = pd.DataFrame([caracterizar_peptideo(s) for s in df_neg_all["sequence"]])
    props_pos_all = pd.DataFrame([caracterizar_peptideo(s) for s in df_pos_all["sequence"]])

    print(f"\n=== [3/3] INSPEÇÃO BIOLÓGICA: TOP {top_k} FALSOS POSITIVOS (NON-ABP COM MAIOR SCORE) ===")
    print(df_top_fp.to_string(index=False))

    print("\n--- Comparativo Biofísico Médio (Média ± DP) ---")
    comparativo = pd.DataFrame({
        "Grupo": [
            f"Positivos Reais PGAT (n={len(df_pos_all)})",
            f"Top {top_k} Falsos Positivos PGAT",
            f"Negativos Reais PGAT (n={len(df_neg_all)})",
        ],
        "Carga_pH7.4": [
            f"{props_pos_all['Carga_pH7.4'].mean():+.2f} ± {props_pos_all['Carga_pH7.4'].std():.2f}",
            f"{props['Carga_pH7.4'].mean():+.2f} ± {props['Carga_pH7.4'].std():.2f}",
            f"{props_neg_all['Carga_pH7.4'].mean():+.2f} ± {props_neg_all['Carga_pH7.4'].std():.2f}",
        ],
        "Momento_Hidro_uH": [
            f"{props_pos_all['Momento_Hidro_uH'].mean():.3f} ± {props_pos_all['Momento_Hidro_uH'].std():.3f}",
            f"{props['Momento_Hidro_uH'].mean():.3f} ± {props['Momento_Hidro_uH'].std():.3f}",
            f"{props_neg_all['Momento_Hidro_uH'].mean():.3f} ± {props_neg_all['Momento_Hidro_uH'].std():.3f}",
        ],
        "GRAVY": [
            f"{props_pos_all['GRAVY'].mean():+.3f} ± {props_pos_all['GRAVY'].std():.3f}",
            f"{props['GRAVY'].mean():+.3f} ± {props['GRAVY'].std():.3f}",
            f"{props_neg_all['GRAVY'].mean():+.3f} ± {props_neg_all['GRAVY'].std():.3f}",
        ],
        "Pct_Cationicos_KR": [
            f"{props_pos_all['Pct_Cationicos_KR'].mean():.1f}%",
            f"{props['Pct_Cationicos_KR'].mean():.1f}%",
            f"{props_neg_all['Pct_Cationicos_KR'].mean():.1f}%",
        ],
    })
    print(comparativo.to_string(index=False))

    # Gráfico comparando a distribuição de Carga Líquida e Momento Hidrofóbico
    PASTA_FIG.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    df_plot = pd.concat([
        props_pos_all.assign(Grupo="Positivos Reais (ABP)"),
        props_neg_all.assign(Grupo="Negativos Reais (Non-ABP)"),
        props.assign(Grupo=f"Top {top_k} Falsos Positivos"),
    ], ignore_index=True)

    paleta = {
        "Positivos Reais (ABP)": "#2ca02c",
        "Negativos Reais (Non-ABP)": "#7f7f7f",
        f"Top {top_k} Falsos Positivos": "#d62728",
    }

    sns.boxplot(data=df_plot, x="Grupo", y="Carga_pH7.4", palette=paleta, ax=axes[0], width=0.45)
    axes[0].set(title="Carga Líquida em pH 7.4", xlabel="", ylabel="Carga Líquida")
    axes[0].tick_params(axis="x", rotation=12)

    sns.boxplot(data=df_plot, x="Grupo", y="Momento_Hidro_uH", palette=paleta, ax=axes[1], width=0.45)
    axes[1].set(title="Anfipaticidade (Momento Hidrofóbico μH)", xlabel="", ylabel="μH (Eisenberg, 100°)")
    axes[1].tick_params(axis="x", rotation=12)

    fig.tight_layout()
    caminho_fig = PASTA_FIG / "05_diagnostico_biofisico_fp_pgat.png"
    fig.savefig(caminho_fig, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\n📊 Gráfico biofísico salvo em: {caminho_fig.relative_to(ROOT)}")

    return df_top_fp


# ==========================================
# Execução Principal
# ==========================================
def main() -> None:
    df_pgat = pd.read_csv(PATH_CSV_PGAT)
    X_fq = pd.read_csv(PATH_X_FQ)
    X_esm = pd.read_csv(PATH_X_ESM)
    y_pgat = pd.read_csv(PATH_Y).squeeze("columns").values

    modelo_fq = joblib.load(MODELS_DIR / "modelo_base_fq_calibrado.joblib")
    modelo_esm = joblib.load(MODELS_DIR / "modelo_base_esm2.joblib")
    pacote_meta = joblib.load(MODELS_DIR / "meta_modelo_stacking.joblib")

    meta_modelo = pacote_meta["meta_modelo"]
    limiar_oof = float(pacote_meta["limiar"])

    if hasattr(modelo_fq, "feature_names_in_"):
        X_fq = X_fq[list(modelo_fq.feature_names_in_)]
    if hasattr(modelo_esm, "feature_names_in_"):
        X_esm.columns = modelo_esm.feature_names_in_

    prob_fq = modelo_fq.predict_proba(X_fq)[:, 1]
    prob_esm = modelo_esm.predict_proba(X_esm)[:, 1]
    meta_logits = np.column_stack((
        probs_para_logodds(prob_fq),
        probs_para_logodds(prob_esm),
    ))
    prob_meta = meta_modelo.predict_proba(meta_logits)[:, 1]

    df_shift = avaliar_prior_shift(y_pgat, prob_meta, limiar_oof)
    df_bal = avaliar_subamostragem_balanceada(y_pgat, prob_fq, prob_esm, prob_meta, limiar_oof, n_runs=100)
    df_top_fp = inspecionar_top_falsos_positivos(df_pgat, prob_fq, prob_esm, prob_meta, top_k=10)

    PASTA_TAB.mkdir(parents=True, exist_ok=True)
    df_shift.to_csv(PASTA_TAB / "pgat_analise_prior_shift.csv", index=False)
    df_bal.to_csv(PASTA_TAB / "pgat_subamostragem_balanceada_100runs.csv", index=False)
    df_top_fp.to_csv(PASTA_TAB / "pgat_top10_falsos_positivos.csv", index=False)
    print(f"\n✅ Tabelas salvas em: {PASTA_TAB.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
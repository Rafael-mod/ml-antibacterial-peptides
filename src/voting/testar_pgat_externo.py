"""
src/voting/testar_pgat_externo.py

Avaliação Final do Meta-Modelo (Fusão Tardia) no Benchmark Externo PGAT-ABPp:
1. Carrega 'X_pgat_fq.csv' (5149x73), 'X_pgat_esm2.csv' (5149x1280) e 'y_pgat.csv'.
2. Carrega os modelos treinados em 'models/' sem qualquer re-treino ou ajuste.
3. Calcula previsões individuais (Base A e Base B) e combinadas (Meta-Modelo Log-Odds).
4. Exibe métricas completas, matriz de confusão e auditoria de complementaridade.
5. Exporta tabela (.csv) e figura (.png) em 'reports/'.

Uso:
    python src/voting/testar_pgat_externo.py
"""

from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.special import logit
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    roc_auc_score,
    roc_curve,
)

sns.set_theme(style="whitegrid", context="notebook")

# ==========================================
# Caminhos
# ==========================================
ROOT = Path(__file__).resolve().parents[2]
EXT_DIR = ROOT / "data" / "external" / "pgat"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
PASTA_FIG = REPORTS_DIR / "figuras_modelagem"
PASTA_TAB = REPORTS_DIR / "tabelas"

PATH_X_FQ = EXT_DIR / "X_pgat_fq.csv"
PATH_X_ESM = EXT_DIR / "X_pgat_esm2.csv"
PATH_Y = EXT_DIR / "y_pgat.csv"


# ==========================================
# Funções Auxiliares
# ==========================================
def probs_para_logodds(probs: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return logit(np.clip(probs, eps, 1.0 - eps))


def avaliar_modelo(y_true: np.ndarray, probs: np.ndarray, nome: str, limiar: float) -> dict:
    y_pred = (probs >= limiar).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    mcc = matthews_corrcoef(y_true, y_pred)
    auroc = roc_auc_score(y_true, probs)
    aucpr = average_precision_score(y_true, probs)
    sens = tp / (tp + fn)
    esp = tn / (tn + fp)
    prec = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(
        f"{nome:.<34} | MCC: {mcc:.3f} | AUROC: {auroc:.3f} | AUCPR: {aucpr:.3f} | "
        f"Sens: {sens:.3f} | Esp: {esp:.3f} | F1: {f1:.3f}"
    )

    return {
        "Modelo": nome,
        "Limiar": round(limiar, 3),
        "MCC": round(mcc, 4),
        "AUROC": round(auroc, 4),
        "AUCPR": round(aucpr, 4),
        "Sensibilidade": round(sens, 4),
        "Especificidade": round(esp, 4),
        "Precisao": round(prec, 4),
        "F1": round(f1, 4),
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
    }


def plotar_resultados_pgat(
    y_true: np.ndarray,
    p_fq: np.ndarray,
    p_esm: np.ndarray,
    p_meta: np.ndarray,
    limiar_oof: float,
) -> None:
    PASTA_FIG.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))

    curvas = [
        ("Base A (FQ LightGBM)", p_fq, "#1f77b4", "--"),
        ("Base B (ESM-2 Linear)", p_esm, "#ff7f0e", "-."),
        ("Meta-Modelo (Fusão Log-Odds)", p_meta, "#2ca02c", "-"),
    ]

    # Painel 1: Curva AUROC
    for nome, probs, cor, ls in curvas:
        fpr, tpr, _ = roc_curve(y_true, probs)
        axes[0].plot(
            fpr, tpr, color=cor, linestyle=ls, lw=2.2,
            label=f"{nome} (AUC = {roc_auc_score(y_true, probs):.3f})",
        )
    axes[0].plot([0, 1], [0, 1], "k:", lw=1, alpha=0.6)
    axes[0].set(
        title="Curva AUROC — Benchmark PGAT-ABPp",
        xlabel="1 − Especificidade (FPR)",
        ylabel="Sensibilidade (TPR)",
    )
    axes[0].legend(loc="lower right", fontsize=8.5)

    # Painel 2: Curva AUCPR
    prevalencia = float(y_true.mean())
    for nome, probs, cor, ls in curvas:
        prec, rec, _ = precision_recall_curve(y_true, probs)
        axes[1].plot(
            rec, prec, color=cor, linestyle=ls, lw=2.2,
            label=f"{nome} (AUCPR = {average_precision_score(y_true, probs):.3f})",
        )
    axes[1].axhline(
        prevalencia, color="gray", linestyle=":", lw=1.2,
        label=f"Baseline Aleatório ({prevalencia:.3f})",
    )
    axes[1].set(
        title="Curva AUCPR — Benchmark PGAT-ABPp",
        xlabel="Recall (Sensibilidade)",
        ylabel="Precisão",
        ylim=(0.0, 1.03),
    )
    axes[1].legend(loc="lower left", fontsize=8.5)

    # Painel 3: Matriz de Confusão do Meta-Modelo (Corte OOF)
    y_pred_meta = (p_meta >= limiar_oof).astype(int)
    cm = confusion_matrix(y_true, y_pred_meta)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Greens",
        cbar=False,
        xticklabels=["Prev. Non-ABP (0)", "Prev. ABP (1)"],
        yticklabels=["Real Non-ABP (0)", "Real ABP (1)"],
        ax=axes[2],
    )
    axes[2].set_title(f"Matriz de Confusão — Fusão (Corte {limiar_oof:.2f})")

    fig.tight_layout()
    caminho_fig = PASTA_FIG / "04_benchmark_externo_pgat.png"
    fig.savefig(caminho_fig, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\n📊 Gráfico salvo em: {caminho_fig.relative_to(ROOT)}")


# ==========================================
# Execução Principal
# ==========================================
def main() -> None:
    print("=== [1/3] Carregando Matrizes Extraídas do PGAT-ABPp ===")
    for arq in (PATH_X_FQ, PATH_X_ESM, PATH_Y):
        if not arq.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {arq}")

    X_fq = pd.read_csv(PATH_X_FQ)
    X_esm = pd.read_csv(PATH_X_ESM)
    y_pgat = pd.read_csv(PATH_Y).squeeze("columns").values

    assert len(X_fq) == len(X_esm) == len(y_pgat), "Divergência no número de amostras!"
    print(f"  -> Amostras carregadas: {len(y_pgat)} ({(y_pgat == 1).sum()} ABP | {(y_pgat == 0).sum()} Non-ABP)")

    print("\n=== [2/3] Carregando Modelos Treinados e Executando Inferência ===")
    modelo_fq = joblib.load(MODELS_DIR / "modelo_base_fq_calibrado.joblib")
    modelo_esm = joblib.load(MODELS_DIR / "modelo_base_esm2.joblib")
    pacote_meta = joblib.load(MODELS_DIR / "meta_modelo_stacking.joblib")

    meta_modelo = pacote_meta["meta_modelo"]
    limiar_oof = float(pacote_meta["limiar"])

    # Garante paridade exata dos nomes das colunas esperados pelos modelos
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

    print("\n=== [3/3] RESULTADOS NO BENCHMARK EXTERNO PGAT-ABPp (5.149 PEPTÍDEOS) ===")
    resultados = [
        avaliar_modelo(y_pgat, prob_fq, "Base A (FQ LightGBM Calibrado)", limiar=0.50),
        avaliar_modelo(y_pgat, prob_esm, "Base B (ESM-2 Linear L2)", limiar=0.50),
        avaliar_modelo(y_pgat, prob_meta, "Meta-Modelo (Corte 0.50)", limiar=0.50),
        avaliar_modelo(y_pgat, prob_meta, f"Meta-Modelo (Corte OOF {limiar_oof:.2f})", limiar=limiar_oof),
    ]

    df_res = pd.DataFrame(resultados)
    print("\nDetalhamento da Matriz de Confusão:")
    print(df_res[["Modelo", "Limiar", "TP", "FN", "TN", "FP", "MCC"]].to_string(index=False))

    # Auditoria de Complementaridade
    acertou_fq = (prob_fq >= 0.50) == y_pgat
    acertou_esm = (prob_esm >= 0.50) == y_pgat
    acertou_meta = (prob_meta >= limiar_oof) == y_pgat

    print("\n====== AUDITORIA DE COMPLEMENTARIDADE (PGAT-ABPp) ======")
    print(f"  -> Ambos acertaram:                    {(acertou_fq & acertou_esm).sum()}")
    print(f"  -> Só FQ acertou (resgatados Meta):    {(acertou_fq & ~acertou_esm & acertou_meta).sum()}/{(acertou_fq & ~acertou_esm).sum()}")
    print(f"  -> Só ESM-2 acertou (resgatados Meta): {(~acertou_fq & acertou_esm & acertou_meta).sum()}/{(~acertou_fq & acertou_esm).sum()}")
    print(f"  -> Ambos erraram:                      {(~acertou_fq & ~acertou_esm).sum()}")

    PASTA_TAB.mkdir(parents=True, exist_ok=True)
    caminho_tab = PASTA_TAB / "resultados_benchmark_pgat.csv"
    df_res.to_csv(caminho_tab, index=False)
    print(f"\n📄 Tabela salva em: {caminho_tab.relative_to(ROOT)}")

    plotar_resultados_pgat(y_pgat, prob_fq, prob_esm, prob_meta, limiar_oof)


if __name__ == "__main__":
    main()
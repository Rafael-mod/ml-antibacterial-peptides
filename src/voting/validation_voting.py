"""
src/voting/validar_10_runs.py

Validação Rigorosa em 10 Execuções (10x 5-Fold Nested Stratified CV):
1. Roda 10 seeds distintas estritamente no conjunto de treino (4.104 amostras).
2. Treina o Meta-Modelo e otimiza o corte em dobras internas (zero vazamento).
3. Calcula Média ± Desvio Padrão de MCC, AUROC, AUCPR, Sensibilidade e Especificidade.
4. Gera as curvas médias AUROC e AUCPR (± 1 DP) comparando Base A, Base B e Fusão.
"""

import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import logit
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sns.set_theme(style="whitegrid", context="notebook")

# ==========================================
# Configurações e Caminhos
# ==========================================
ROOT = Path(__file__).resolve().parents[2]
PROC_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
PASTA_FIG = REPORTS_DIR / "figuras_modelagem"
PASTA_TAB = REPORTS_DIR / "tabelas"

SEEDS = [42, 100, 202, 303, 404, 505, 606, 707, 808, 909]
OUTER_SPLITS = 5
INNER_SPLITS = 4


def probs_para_logodds(probs: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return logit(np.clip(probs, eps, 1.0 - eps))


def encontrar_melhor_limiar(y_true: np.ndarray, probs: np.ndarray) -> float:
    limiares = np.linspace(0.30, 0.70, 81)
    melhor_t, melhor_mcc = 0.50, -1.0
    for t in limiares:
        mcc = matthews_corrcoef(y_true, (probs >= t).astype(int))
        if mcc > melhor_mcc:
            melhor_mcc, melhor_t = mcc, float(t)
    return melhor_t


def calcular_metricas(y_true: np.ndarray, probs: np.ndarray, limiar: float = 0.50) -> dict:
    y_pred = (probs >= limiar).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "mcc": matthews_corrcoef(y_true, y_pred),
        "auroc": roc_auc_score(y_true, probs),
        "aucpr": average_precision_score(y_true, probs),
        "sens": tp / (tp + fn),
        "esp": tn / (tn + fp),
    }


def executar_run_nested_cv(modelo_fq_base, X_fq: pd.DataFrame, X_esm: pd.DataFrame, y: np.ndarray, seed: int):
    """Executa 1 rodada completa de 5-Fold Nested CV para uma semente específica."""
    n = len(y)
    oof_fq = np.zeros(n)
    oof_esm = np.zeros(n)
    oof_meta = np.zeros(n)
    oof_meta_pred_otimo = np.zeros(n, dtype=int)

    outer_cv = StratifiedKFold(n_splits=OUTER_SPLITS, shuffle=True, random_state=seed)

    for idx_tr, idx_val in outer_cv.split(X_fq, y):
        X_fq_tr, X_fq_val = X_fq.iloc[idx_tr], X_fq.iloc[idx_val]
        X_esm_tr, X_esm_val = X_esm.iloc[idx_tr], X_esm.iloc[idx_val]
        y_tr_fold = y[idx_tr]

        # Modelos base parametrizados com a seed da rodada
        calib_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
        base_fq = CalibratedClassifierCV(estimator=clone(modelo_fq_base), method="sigmoid", cv=calib_cv)
        base_esm = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.05, max_iter=1000, random_state=seed),
        )

        # 1) OOF interno (apenas dentro de idx_tr) para treinar o Meta-Modelo sem vazamento
        inner_probs = np.zeros((len(y_tr_fold), 2))
        inner_cv = StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=seed)

        for in_tr, in_val in inner_cv.split(X_fq_tr, y_tr_fold):
            m_fq_in = clone(base_fq).fit(X_fq_tr.iloc[in_tr], y_tr_fold[in_tr])
            m_esm_in = clone(base_esm).fit(X_esm_tr.iloc[in_tr], y_tr_fold[in_tr])

            inner_probs[in_val, 0] = m_fq_in.predict_proba(X_fq_tr.iloc[in_val])[:, 1]
            inner_probs[in_val, 1] = m_esm_in.predict_proba(X_esm_tr.iloc[in_val])[:, 1]

        meta_X_in = np.column_stack((
            probs_para_logodds(inner_probs[:, 0]),
            probs_para_logodds(inner_probs[:, 1]),
        ))
        meta_clf = LogisticRegression(C=0.5, random_state=seed).fit(meta_X_in, y_tr_fold)
        limiar_in = encontrar_melhor_limiar(y_tr_fold, meta_clf.predict_proba(meta_X_in)[:, 1])

        # 2) Ajusta modelos base no treino externo completo e prevê a dobra externa (idx_val)
        m_fq_out = clone(base_fq).fit(X_fq_tr, y_tr_fold)
        m_esm_out = clone(base_esm).fit(X_esm_tr, y_tr_fold)

        p_fq_val = m_fq_out.predict_proba(X_fq_val)[:, 1]
        p_esm_val = m_esm_out.predict_proba(X_esm_val)[:, 1]

        meta_X_val = np.column_stack((
            probs_para_logodds(p_fq_val),
            probs_para_logodds(p_esm_val),
        ))
        p_meta_val = meta_clf.predict_proba(meta_X_val)[:, 1]

        oof_fq[idx_val] = p_fq_val
        oof_esm[idx_val] = p_esm_val
        oof_meta[idx_val] = p_meta_val
        oof_meta_pred_otimo[idx_val] = (p_meta_val >= limiar_in).astype(int)

    return oof_fq, oof_esm, oof_meta, oof_meta_pred_otimo


def plotar_curvas_medias(y: np.ndarray, historico_probs: dict, historico_metricas: dict) -> None:
    """Gera figura com Curva AUROC média, Curva AUCPR média e distribuição do MCC."""
    PASTA_FIG.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))

    grid_x = np.linspace(0, 1, 300)
    estilos = {
        "Base A (FQ LightGBM)": {"cor": "#1f77b4", "ls": "--"},
        "Base B (ESM-2 Linear)": {"cor": "#ff7f0e", "ls": "-."},
        "Meta-Modelo (Fusão Log-Odds)": {"cor": "#2ca02c", "ls": "-"},
    }

    # --- Painel 1: Curva AUROC Média ---
    for nome, cfg in estilos.items():
        tprs = []
        for probs in historico_probs[nome]:
            fpr, tpr, _ = roc_curve(y, probs)
            interp_tpr = np.interp(grid_x, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        std_tpr = np.std(tprs, axis=0)

        auc_m = np.mean([m["auroc"] for m in historico_metricas[nome]])
        auc_s = np.std([m["auroc"] for m in historico_metricas[nome]])

        axes[0].plot(
            grid_x, mean_tpr, color=cfg["cor"], linestyle=cfg["ls"], lw=2.2,
            label=f"{nome}\n(AUROC = {auc_m:.3f} ± {auc_s:.3f})"
        )
        axes[0].fill_between(
            grid_x, np.clip(mean_tpr - std_tpr, 0, 1), np.clip(mean_tpr + std_tpr, 0, 1),
            color=cfg["cor"], alpha=0.14
        )

    axes[0].plot([0, 1], [0, 1], "k:", lw=1, alpha=0.6)
    axes[0].set(
        title="Curva AUROC Média (10 Runs Nested CV)",
        xlabel="Taxa de Falsos Positivos (1 − Especificidade)",
        ylabel="Taxa de Verdadeiros Positivos (Sensibilidade)",
    )
    axes[0].legend(loc="lower right", fontsize=8.5)

    # --- Painel 2: Curva AUCPR Média ---
    for nome, cfg in estilos.items():
        precs = []
        for probs in historico_probs[nome]:
            prec, rec, _ = precision_recall_curve(y, probs)
            # Inverte arrays pois precision_recall_curve retorna recall decrescente
            interp_prec = np.interp(grid_x, rec[::-1], prec[::-1])
            precs.append(interp_prec)

        mean_prec = np.mean(precs, axis=0)
        std_prec = np.std(precs, axis=0)

        pr_m = np.mean([m["aucpr"] for m in historico_metricas[nome]])
        pr_s = np.std([m["aucpr"] for m in historico_metricas[nome]])

        axes[1].plot(
            grid_x, mean_prec, color=cfg["cor"], linestyle=cfg["ls"], lw=2.2,
            label=f"{nome}\n(AUCPR = {pr_m:.3f} ± {pr_s:.3f})"
        )
        axes[1].fill_between(
            grid_x, np.clip(mean_prec - std_prec, 0, 1), np.clip(mean_prec + std_prec, 0, 1),
            color=cfg["cor"], alpha=0.14
        )

    axes[1].set(
        title="Curva AUCPR Média (10 Runs Nested CV)",
        xlabel="Recall (Sensibilidade)",
        ylabel="Precisão",
        ylim=(0.45, 1.02),
    )
    axes[1].legend(loc="lower left", fontsize=8.5)

    # --- Painel 3: Estabilidade do MCC nas 10 Seeds ---
    df_mcc = pd.DataFrame({
        "Base A\n(FQ)": [m["mcc"] for m in historico_metricas["Base A (FQ LightGBM)"]],
        "Base B\n(ESM-2)": [m["mcc"] for m in historico_metricas["Base B (ESM-2 Linear)"]],
        "Fusão\n(Corte 0.50)": [m["mcc"] for m in historico_metricas["Meta-Modelo (Fusão Log-Odds)"]],
        "Fusão\n(Corte OOF)": [m["mcc"] for m in historico_metricas["Meta-Modelo (Corte OOF)"]],
    })
    sns.boxplot(data=df_mcc, ax=axes[2], palette=["#1f77b4", "#ff7f0e", "#2ca02c", "#1b7837"], width=0.45)
    sns.stripplot(data=df_mcc, ax=axes[2], color="black", size=5, alpha=0.75)
    axes[2].set(title="Estabilidade do MCC (10 Seeds)", ylabel="Matthews Correlation Coefficient (MCC)")

    fig.tight_layout()
    caminho_fig = PASTA_FIG / "03_validacao_10_runs_auroc_aucpr.png"
    fig.savefig(caminho_fig, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\n📊 Gráfico salvo em: {caminho_fig.relative_to(ROOT)}")


def main() -> None:
    print("=== Validação Rigorosa: 10 Runs x 5-Fold Nested CV (Apenas Treino) ===")
    X_fq = pd.read_csv(PROC_DIR / "X_train_limpo.csv")
    X_esm = pd.read_csv(PROC_DIR / "X_train_esm2.csv")
    y = pd.read_csv(PROC_DIR / "y_train.csv").squeeze("columns").values
    modelo_fq_base = joblib.load(MODELS_DIR / "modelo_final.joblib")

    chaves = [
        "Base A (FQ LightGBM)",
        "Base B (ESM-2 Linear)",
        "Meta-Modelo (Fusão Log-Odds)",
        "Meta-Modelo (Corte OOF)",
    ]
    historico_metricas = {k: [] for k in chaves}
    historico_probs = {k: [] for k in chaves[:3]}

    t_inicio = time.time()
    for i, seed in enumerate(SEEDS, 1):
        t0 = time.time()
        oof_fq, oof_esm, oof_meta, oof_pred_otimo = executar_run_nested_cv(
            modelo_fq_base, X_fq, X_esm, y, seed
        )

        m_fq = calcular_metricas(y, oof_fq, limiar=0.50)
        m_esm = calcular_metricas(y, oof_esm, limiar=0.50)
        m_meta = calcular_metricas(y, oof_meta, limiar=0.50)

        # Métricas do corte OOF dinâmico por dobra
        tn, fp, fn, tp = confusion_matrix(y, oof_pred_otimo).ravel()
        m_meta_otimo = {
            "mcc": matthews_corrcoef(y, oof_pred_otimo),
            "auroc": m_meta["auroc"],
            "aucpr": m_meta["aucpr"],
            "sens": tp / (tp + fn),
            "esp": tn / (tn + fp),
        }

        historico_metricas["Base A (FQ LightGBM)"].append(m_fq)
        historico_metricas["Base B (ESM-2 Linear)"].append(m_esm)
        historico_metricas["Meta-Modelo (Fusão Log-Odds)"].append(m_meta)
        historico_metricas["Meta-Modelo (Corte OOF)"].append(m_meta_otimo)

        historico_probs["Base A (FQ LightGBM)"].append(oof_fq)
        historico_probs["Base B (ESM-2 Linear)"].append(oof_esm)
        historico_probs["Meta-Modelo (Fusão Log-Odds)"].append(oof_meta)

        print(
            f"  [Run {i:02d}/10 | Seed {seed:<3}] "
            f"MCC FQ: {m_fq['mcc']:.3f} | MCC ESM: {m_esm['mcc']:.3f} | "
            f"MCC Fusão: {m_meta['mcc']:.3f} (OOF: {m_meta_otimo['mcc']:.3f}) | "
            f"AUROC: {m_meta['auroc']:.4f} ({time.time() - t0:.1f}s)"
        )

    # Consolida tabela final (Média ± DP)
    linhas_resumo = []
    for nome in chaves:
        df_m = pd.DataFrame(historico_metricas[nome])
        linhas_resumo.append({
            "Modelo": nome,
            "MCC": f"{df_m['mcc'].mean():.3f} ± {df_m['mcc'].std():.3f}",
            "AUROC": f"{df_m['auroc'].mean():.4f} ± {df_m['auroc'].std():.4f}",
            "AUCPR": f"{df_m['aucpr'].mean():.4f} ± {df_m['aucpr'].std():.4f}",
            "Sensibilidade": f"{df_m['sens'].mean():.3f} ± {df_m['sens'].std():.3f}",
            "Especificidade": f"{df_m['esp'].mean():.3f} ± {df_m['esp'].std():.3f}",
        })

    df_final = pd.DataFrame(linhas_resumo)
    PASTA_TAB.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(PASTA_TAB / "validacao_10_runs_nested_cv.csv", index=False)

    print(f"\n====== RESUMO DAS 10 EXECUÇÕES (MÉDIA ± DESVIO PADRÃO) [{time.time() - t_inicio:.1f}s] ======")
    print(df_final.to_string(index=False))

    plotar_curvas_medias(y, historico_probs, historico_metricas)


if __name__ == "__main__":
    main()
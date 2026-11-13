"""
src/models/late_fusion.py

Arquitetura de Fusão Tardia (Stacking Ensemble Otimizado):
1. Modelo Base A: LightGBM Calibrado (73 Features Físico-Químicas).
2. Modelo Base B: Regressão Logística L2 Padronizada (1280 Embeddings ESM-2).
3. Meta-Modelo: Regressão Logística no espaço Log-Odds + Otimização de Limiar (MCC).
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. Configurações
# ==========================================
ROOT = Path(__file__).resolve().parents[2]
PROC_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
SEED = 42

PATH_MODELO_FQ = MODELS_DIR / "modelo_final.joblib"
PATH_X_TR_FQ = PROC_DIR / "X_train_limpo.csv"
PATH_X_TE_FQ = PROC_DIR / "X_test_limpo.csv"
PATH_X_TR_ESM = PROC_DIR / "X_train_esm2.csv"
PATH_X_TE_ESM = PROC_DIR / "X_test_esm2.csv"
PATH_Y_TR = PROC_DIR / "y_train.csv"
PATH_Y_TE = PROC_DIR / "y_test.csv"


def probs_para_logodds(probs: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Converte probabilidades [0, 1] para escala log-odds (-inf, +inf)."""
    p_clip = np.clip(probs, eps, 1.0 - eps)
    return logit(p_clip)


# ==========================================
# 2. Carregamento
# ==========================================
def carregar_dados():
    print("Carregando dados e Modelo FQ (LightGBM)...")
    X_tr_fq = pd.read_csv(PATH_X_TR_FQ)
    X_te_fq = pd.read_csv(PATH_X_TE_FQ)
    X_tr_esm = pd.read_csv(PATH_X_TR_ESM)
    X_te_esm = pd.read_csv(PATH_X_TE_ESM)

    y_tr = pd.read_csv(PATH_Y_TR).squeeze("columns").values
    y_te = pd.read_csv(PATH_Y_TE).squeeze("columns").values

    modelo_fq = joblib.load(PATH_MODELO_FQ)
    return X_tr_fq, X_te_fq, X_tr_esm, X_te_esm, y_tr, y_te, modelo_fq


# ==========================================
# 3. Geração OOF (Out-of-Fold) Calibrada
# ==========================================
def gerar_probs_meta(modelo_fq, X_tr_fq, X_tr_esm, y_tr, n_splits=5):
    print("\nGerando previsões Out-of-Fold Calibradas (5-Fold CV)...")

    oof_probs = np.zeros((len(y_tr), 2))  # Col 0: FQ calibrado, Col 1: ESM-2

    # Base A encapsulada com calibração sigmoide (Platt scaling)
    base_fq_calibrada = CalibratedClassifierCV(
        estimator=clone(modelo_fq), method="sigmoid", cv=3
    )

    # Base B: Modelo linear regularizado L2 (ideal para embeddings contínuos 1280-d)
    base_esm = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.05, max_iter=1000, random_state=SEED),
    )

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    for fold, (idx_tr, idx_val) in enumerate(kf.split(X_tr_fq, y_tr)):
        print(f"  Fold {fold + 1}/{n_splits}...")

        # Base A (FQ Calibrado)
        mod_fq_fold = clone(base_fq_calibrada)
        mod_fq_fold.fit(X_tr_fq.iloc[idx_tr], y_tr[idx_tr])
        oof_probs[idx_val, 0] = mod_fq_fold.predict_proba(X_tr_fq.iloc[idx_val])[:, 1]

        # Base B (ESM-2 Linear L2)
        mod_esm_fold = clone(base_esm)
        mod_esm_fold.fit(X_tr_esm.iloc[idx_tr], y_tr[idx_tr])
        oof_probs[idx_val, 1] = mod_esm_fold.predict_proba(X_tr_esm.iloc[idx_val])[:, 1]

    print("\nAjustando modelos base finais na base completa de treino...")
    modelo_fq_final = clone(base_fq_calibrada).fit(X_tr_fq, y_tr)
    modelo_esm_final = clone(base_esm).fit(X_tr_esm, y_tr)

    return oof_probs, modelo_fq_final, modelo_esm_final


# ==========================================
# 4. Otimização de Limiar e Avaliação
# ==========================================
def encontrar_melhor_limiar(y_true: np.ndarray, probs: np.ndarray) -> float:
    limiares = np.linspace(0.30, 0.70, 81)
    melhor_t, melhor_mcc = 0.50, -1.0
    for t in limiares:
        mcc = matthews_corrcoef(y_true, (probs >= t).astype(int))
        if mcc > melhor_mcc:
            melhor_mcc, melhor_t = mcc, float(t)
    return melhor_t


def avaliar_modelo(y_true, probs, nome, limiar=0.50):
    y_pred = (probs >= limiar).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    mcc = matthews_corrcoef(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, probs)
    pr_auc = average_precision_score(y_true, probs)
    sens = tp / (tp + fn)
    esp = tn / (tn + fp)

    print(
        f"{nome:.<32} | MCC: {mcc:.3f} | AUC: {roc_auc:.3f} | "
        f"PR-AUC: {pr_auc:.3f} | Sens: {sens:.3f} | Esp: {esp:.3f}"
    )


# ==========================================
# 5. Execução Principal
# ==========================================
def main():
    X_tr_fq, X_te_fq, X_tr_esm, X_te_esm, y_tr, y_te, modelo_fq_raw = carregar_dados()

    # 1. Gera previsões OOF calibradas e ajusta modelos finais
    oof_probs, modelo_fq_final, modelo_esm_final = gerar_probs_meta(
        modelo_fq_raw, X_tr_fq, X_tr_esm, y_tr
    )

    # Converte OOF para espaço Log-Odds
    meta_tr_logits = np.column_stack((
        probs_para_logodds(oof_probs[:, 0]),
        probs_para_logodds(oof_probs[:, 1]),
    ))

    # 2. Treina Meta-Modelo regularizado sobre os Log-Odds
    print("\nTreinando Meta-Modelo (Regressão Logística em Log-Odds)...")
    meta_modelo = LogisticRegression(C=0.5, random_state=SEED)
    meta_modelo.fit(meta_tr_logits, y_tr)

    coef_fq, coef_esm = meta_modelo.coef_[0]
    print(f"  -> Peso Log-Odds FQ (LightGBM):  {coef_fq:.3f}")
    print(f"  -> Peso Log-Odds ESM-2 (Linear): {coef_esm:.3f}")

    # Limiar ótimo encontrado exclusivamente no treino (OOF)
    oof_meta_probs = meta_modelo.predict_proba(meta_tr_logits)[:, 1]
    limiar_otimo = encontrar_melhor_limiar(y_tr, oof_meta_probs)
    print(f"  -> Limiar ótimo OOF (MCC):       {limiar_otimo:.3f}")

    # 3. Previsões no Teste Cego
    prob_te_fq = modelo_fq_final.predict_proba(X_te_fq)[:, 1]
    prob_te_esm = modelo_esm_final.predict_proba(X_te_esm)[:, 1]

    meta_te_logits = np.column_stack((
        probs_para_logodds(prob_te_fq),
        probs_para_logodds(prob_te_esm),
    ))
    probs_finais = meta_modelo.predict_proba(meta_te_logits)[:, 1]

    print("\n====== RESULTADOS NO TESTE CEGO (ALINHADO) ======")
    avaliar_modelo(y_te, prob_te_fq, "Base A (FQ LightGBM Calibrado)", limiar=0.50)
    avaliar_modelo(y_te, prob_te_esm, "Base B (ESM-2 Reg. Logística L2)", limiar=0.50)
    avaliar_modelo(y_te, probs_finais, "Meta-Modelo (Corte 0.50)", limiar=0.50)
    avaliar_modelo(y_te, probs_finais, f"Meta-Modelo (Corte OOF {limiar_otimo:.2f})", limiar=limiar_otimo)

    # 4. Auditoria de Complementaridade Real no Teste Cego
    acertou_fq = (prob_te_fq >= 0.50) == y_te
    acertou_esm = (prob_te_esm >= 0.50) == y_te
    acertou_meta = (probs_finais >= limiar_otimo) == y_te

    print("\n====== AUDITORIA DE COMPLEMENTARIDADE (TESTE CEGO) ======")
    print(f"  -> Ambos acertaram:                   {(acertou_fq & acertou_esm).sum()}")
    print(f"  -> Só FQ acertou (resgatados Meta):   {(acertou_fq & ~acertou_esm & acertou_meta).sum()}/{(acertou_fq & ~acertou_esm).sum()}")
    print(f"  -> Só ESM-2 acertou (resgatados Meta):{(~acertou_fq & acertou_esm & acertou_meta).sum()}/{(~acertou_fq & acertou_esm).sum()}")
    print(f"  -> Ambos erraram:                     {(~acertou_fq & ~acertou_esm).sum()}")

    # 5. Salva Modelos
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(modelo_fq_final, MODELS_DIR / "modelo_base_fq_calibrado.joblib")
    joblib.dump(modelo_esm_final, MODELS_DIR / "modelo_base_esm2.joblib")
    joblib.dump({"meta_modelo": meta_modelo, "limiar": limiar_otimo}, MODELS_DIR / "meta_modelo_stacking.joblib")
    print(f"\n✅ Modelos atualizados salvos em {MODELS_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
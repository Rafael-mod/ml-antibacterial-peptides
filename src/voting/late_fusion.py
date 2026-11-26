"""
src/models/late_fusion.py

Arquitetura de Fusão Tardia (Stacking Ensemble Otimizado):
1. Modelo Base A: LightGBM Calibrado (148 Features Físico-Químicas).
2. Modelo Base B: ESM-2 (650M) + Gated Multi-Head Attention Pooling (PyTorch + Temp. Scaling).
3. Meta-Modelo: Regressão Logística no espaço Log-Odds + Otimização de Limiar (MCC).
"""

from pathlib import Path
import copy
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    log_loss,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset

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
PATH_PT_TR_ESM = PROC_DIR / "esm2_train_residuos.pt"
PATH_PT_TE_ESM = PROC_DIR / "esm2_test_residuos.pt"
PATH_Y_TR = PROC_DIR / "y_train.csv"
PATH_Y_TE = PROC_DIR / "y_test.csv"

# Hiperparâmetros da Rede de Attention Pooling
ARCH_PARAMS = {
    "emb_dim": 1280,
    "d_model": 256,
    "num_heads": 4,
    "dropout": 0.35,
}
BATCH_SIZE = 32
EPOCHS = 35
PATIENCE = 7
LR = 3e-4
WEIGHT_DECAY = 2e-2


def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def probs_para_logodds(probs: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Converte probabilidades [0, 1] para escala log-odds (-inf, +inf)."""
    p_clip = np.clip(probs, eps, 1.0 - eps)
    return logit(p_clip)


# ==========================================
# 2. Arquitetura PyTorch: Attention Pooling
# ==========================================
class GatedMultiHeadAttentionPooling(nn.Module):
    def __init__(self, d_model: int = 256, num_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.attn_V = nn.Linear(d_model, d_model)
        self.attn_U = nn.Linear(d_model, d_model)
        self.attn_weights = nn.Linear(d_model, num_heads)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        gated = torch.tanh(self.attn_V(x)) * torch.sigmoid(self.attn_U(x))
        scores = self.attn_weights(self.dropout(gated)).transpose(1, 2)  # (B, H, L)
        scores = scores.masked_fill(~mask.unsqueeze(1), -1e9)

        alpha = F.softmax(scores, dim=-1)  # (B, H, L)
        pooled = torch.bmm(alpha, x).reshape(x.size(0), -1)  # (B, H * D)
        return pooled, alpha


class ESM2AttentionClassifier(nn.Module):
    def __init__(
        self,
        emb_dim: int = 1280,
        d_model: int = 256,
        num_heads: int = 4,
        dropout: float = 0.35,
    ):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(emb_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.local_conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.pool = GatedMultiHeadAttentionPooling(
            d_model=d_model, num_heads=num_heads, dropout=0.2
        )
        pooled_dim = d_model * num_heads
        self.head = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Dropout(dropout),
            nn.Linear(pooled_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        mask_f = mask.unsqueeze(-1).float()
        h = self.proj(x) * mask_f

        # Convolução 1D residual mascarada (impede contaminação do C-terminal por <pad>)
        h_conv = self.local_conv(h.transpose(1, 2)).transpose(1, 2) * mask_f
        h = h + h_conv

        z, alpha = self.pool(h, mask)
        logits = self.head(z).squeeze(-1)
        return logits, alpha


# ==========================================
# 3. Treino, Calibração e Inferência ESM-2
# ==========================================
def prever_logits_esm(
    modelo: nn.Module,
    emb: torch.Tensor,
    mask: torch.Tensor,
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    modelo.eval()
    logits_list = []
    with torch.inference_mode():
        for i in range(0, len(emb), batch_size):
            xb = emb[i : i + batch_size].to(device=device, dtype=torch.float32)
            mb = mask[i : i + batch_size].to(device=device)
            logits, _ = modelo(xb, mb)
            logits_list.append(logits.cpu().numpy())
    return np.concatenate(logits_list)


def treinar_fold_esm_attention(
    emb_tr: torch.Tensor,
    mask_tr: torch.Tensor,
    y_tr: np.ndarray,
    emb_val: torch.Tensor,
    mask_val: torch.Tensor,
    y_val: np.ndarray,
    device: torch.device,
    fold_seed: int,
):
    set_seed(fold_seed)
    modelo = ESM2AttentionClassifier(**ARCH_PARAMS).to(device)
    optimizer = torch.optim.AdamW(modelo.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.BCEWithLogitsLoss()

    ds_tr = TensorDataset(
        emb_tr, mask_tr, torch.tensor(y_tr, dtype=torch.float32)
    )
    loader_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True)

    melhor_auc = -1.0
    melhor_estado = None
    epochs_sem_melhora = 0

    for _ in range(EPOCHS):
        modelo.train()
        for xb, mb, yb in loader_tr:
            xb = xb.to(device=device, dtype=torch.float32)
            mb = mb.to(device=device)
            yb = yb.to(device=device)

            optimizer.zero_grad()
            logits, _ = modelo(xb, mb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()

        val_logits = prever_logits_esm(modelo, emb_val, mask_val, device)
        val_auc = roc_auc_score(y_val, expit(val_logits))

        if val_auc > melhor_auc + 1e-4:
            melhor_auc = val_auc
            melhor_estado = copy.deepcopy(modelo.state_dict())
            epochs_sem_melhora = 0
        else:
            epochs_sem_melhora += 1
            if epochs_sem_melhora >= PATIENCE:
                break

    modelo.load_state_dict(melhor_estado)

    # Calibração via Temperature Scaling no conjunto de validação do fold
    val_logits_final = prever_logits_esm(modelo, emb_val, mask_val, device)
    res_temp = minimize_scalar(
        lambda T: log_loss(y_val, expit(val_logits_final / T)),
        bounds=(0.2, 5.0),
        method="bounded",
    )
    temp_otima = float(res_temp.x)
    oof_probs_val = expit(val_logits_final / temp_otima)

    return oof_probs_val, melhor_estado, temp_otima, melhor_auc


# ==========================================
# 4. Carregamento
# ==========================================
def carregar_dados():
    print("Carregando dados, Modelo FQ (LightGBM) e tensores por resíduo ESM-2...")
    X_tr_fq = pd.read_csv(PATH_X_TR_FQ)
    X_te_fq = pd.read_csv(PATH_X_TE_FQ)

    y_tr = pd.read_csv(PATH_Y_TR).squeeze("columns").values
    y_te = pd.read_csv(PATH_Y_TE).squeeze("columns").values

    pacote_tr = torch.load(PATH_PT_TR_ESM, map_location="cpu", weights_only=False)
    pacote_te = torch.load(PATH_PT_TE_ESM, map_location="cpu", weights_only=False)

    assert (pacote_tr["labels"].numpy().astype(int) == y_tr).all(), "Desalinhamento em esm2_train_residuos.pt!"
    assert (pacote_te["labels"].numpy().astype(int) == y_te).all(), "Desalinhamento em esm2_test_residuos.pt!"

    modelo_fq = joblib.load(PATH_MODELO_FQ)
    return X_tr_fq, X_te_fq, pacote_tr, pacote_te, y_tr, y_te, modelo_fq


# ==========================================
# 5. Geração OOF (Out-of-Fold) Calibrada
# ==========================================
def gerar_probs_meta(modelo_fq, X_tr_fq, pacote_tr, pacote_te, y_tr, n_splits=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nGerando previsões Out-of-Fold Calibradas (5-Fold CV) | Device ESM-2: {device}...")

    oof_probs = np.zeros((len(y_tr), 2))  # Col 0: FQ calibrado, Col 1: ESM-2 Attention
    test_probs_folds = np.zeros((n_splits, len(pacote_te["labels"])))
    esm_fold_checkpoints = []

    base_fq_calibrada = CalibratedClassifierCV(
        estimator=clone(modelo_fq), method="sigmoid", cv=3
    )

    emb_tr_all = pacote_tr["embeddings"]
    mask_tr_all = pacote_tr["mask"]
    emb_te_all = pacote_te["embeddings"]
    mask_te_all = pacote_te["mask"]

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    for fold, (idx_tr, idx_val) in enumerate(kf.split(X_tr_fq, y_tr)):
        # Base A (FQ Calibrado)
        mod_fq_fold = clone(base_fq_calibrada)
        mod_fq_fold.fit(X_tr_fq.iloc[idx_tr], y_tr[idx_tr])
        oof_probs[idx_val, 0] = mod_fq_fold.predict_proba(X_tr_fq.iloc[idx_val])[:, 1]

        # Base B (ESM-2 Attention Pooling + Temperature Scaling)
        oof_val_esm, state_dict, temp, val_auc = treinar_fold_esm_attention(
            emb_tr_all[idx_tr],
            mask_tr_all[idx_tr],
            y_tr[idx_tr],
            emb_tr_all[idx_val],
            mask_tr_all[idx_val],
            y_tr[idx_val],
            device=device,
            fold_seed=SEED + fold,
        )
        oof_probs[idx_val, 1] = oof_val_esm
        esm_fold_checkpoints.append({"state_dict": state_dict, "temperature": temp})

        # Inferência no Teste Cego usando o modelo calibrado deste fold
        mod_eval = ESM2AttentionClassifier(**ARCH_PARAMS).to(device)
        mod_eval.load_state_dict(state_dict)
        logits_te = prever_logits_esm(mod_eval, emb_te_all, mask_te_all, device)
        test_probs_folds[fold] = expit(logits_te / temp)

        print(f"  Fold {fold + 1}/{n_splits} | ESM-2 Val AUROC: {val_auc:.4f} | Temp: {temp:.3f}")

    print("\nAjustando modelo base FQ na base completa e consolidando Ensemble ESM-2 (5 folds)...")
    modelo_fq_final = clone(base_fq_calibrada).fit(X_tr_fq, y_tr)
    prob_te_esm_final = test_probs_folds.mean(axis=0)

    return oof_probs, modelo_fq_final, prob_te_esm_final, esm_fold_checkpoints


# ==========================================
# 6. Otimização de Limiar e Avaliação
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
        f"{nome:.<34} | MCC: {mcc:.3f} | AUC: {roc_auc:.3f} | "
        f"PR-AUC: {pr_auc:.3f} | Sens: {sens:.3f} | Esp: {esp:.3f}"
    )


# ==========================================
# 7. Execução Principal
# ==========================================
def main():
    set_seed(SEED)
    X_tr_fq, X_te_fq, pacote_tr, pacote_te, y_tr, y_te, modelo_fq_raw = carregar_dados()

    # 1. Gera previsões OOF calibradas e ajusta modelos finais
    oof_probs, modelo_fq_final, prob_te_esm, esm_checkpoints = gerar_probs_meta(
        modelo_fq_raw, X_tr_fq, pacote_tr, pacote_te, y_tr
    )

    meta_tr_logits = np.column_stack((
        probs_para_logodds(oof_probs[:, 0]),
        probs_para_logodds(oof_probs[:, 1]),
    ))

    # 2. Treina Meta-Modelo regularizado sobre os Log-Odds
    print("\nTreinando Meta-Modelo (Regressão Logística em Log-Odds)...")
    meta_modelo = LogisticRegression(C=0.5, random_state=SEED)
    meta_modelo.fit(meta_tr_logits, y_tr)

    coef_fq, coef_esm = meta_modelo.coef_[0]
    print(f"  -> Peso Log-Odds FQ (LightGBM):     {coef_fq:.3f}")
    print(f"  -> Peso Log-Odds ESM-2 (Attention): {coef_esm:.3f}")

    limiar_otimo = encontrar_melhor_limiar(y_tr, meta_modelo.predict_proba(meta_tr_logits)[:, 1])
    print(f"  -> Limiar ótimo OOF (MCC):          {limiar_otimo:.3f}")

    # 3. Previsões no Teste Cego
    prob_te_fq = modelo_fq_final.predict_proba(X_te_fq)[:, 1]

    meta_te_logits = np.column_stack((
        probs_para_logodds(prob_te_fq),
        probs_para_logodds(prob_te_esm),
    ))
    probs_finais = meta_modelo.predict_proba(meta_te_logits)[:, 1]

    print("\n====== RESULTADOS NO TESTE CEGO (ALINHADO) ======")
    avaliar_modelo(y_te, prob_te_fq, "Base A (FQ LightGBM Calibrado)", limiar=0.50)
    avaliar_modelo(y_te, prob_te_esm, "Base B (ESM-2 Attention Pooling)", limiar=0.50)
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
    torch.save(
        {
            "arch_params": ARCH_PARAMS,
            "fold_models": esm_checkpoints,
        },
        MODELS_DIR / "modelo_base_esm2_attn.pt",
    )
    joblib.dump(
        {"meta_modelo": meta_modelo, "limiar": limiar_otimo},
        MODELS_DIR / "meta_modelo_stacking.joblib",
    )
    print(f"\n✅ Modelos atualizados salvos em {MODELS_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
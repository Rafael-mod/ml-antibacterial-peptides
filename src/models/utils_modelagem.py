"""
src/models/utils_modelagem.py

Componentes compartilhados entre treinamento.py e tuning.py:
- PreprocessadorDinamico (zero leakage: calcula skew/esparsidade dentro de cada dobra)
- Carregamento de dados respeitando cv_fold / cluster_id (homologia 40%)
- Métricas (MCC, ROC-AUC, PR-AUC, Brier Score)
- Conformal Prediction (Incerteza Calibrada - Caminho C)
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    make_scorer,
    matthews_corrcoef,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    PredefinedSplit,
    StratifiedGroupKFold,
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.preprocessing import PowerTransformer, StandardScaler


# ==========================================
# 1. Transformador Customizado (Zero Leakage)
# ==========================================
class PreprocessadorDinamico(BaseEstimator, TransformerMixin):
    """
    Encapsula a triagem de colunas dentro do fit() de cada dobra da CV:
      - One-Hot (OH_*, OHE_*) ou binárias -> passthrough
      - Esparsas (> limite_esparsidade zeros) ou simétricas -> StandardScaler
      - Contínuas densas assimétricas (|skew| > limite_skew) -> Yeo-Johnson
    """

    def __init__(
        self,
        limite_skew: float = 1.0,
        limite_esparsidade: float = 0.30,
        verbose: bool = False,
    ) -> None:
        self.limite_skew = limite_skew
        self.limite_esparsidade = limite_esparsidade
        self.verbose = verbose

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "PreprocessadorDinamico":
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

        self.oh_binarias_: List[Any] = [
            c for c in df.columns
            if str(c).startswith(("OH_", "OHE_")) or df[c].nunique(dropna=True) <= 2
        ]
        resto = [c for c in df.columns if c not in self.oh_binarias_]

        self.assimetricas_: List[Any] = []
        self.normais_: List[Any] = []

        for col in resto:
            prop_zeros = (df[col] == 0).mean()
            if prop_zeros > self.limite_esparsidade:
                self.normais_.append(col)
            else:
                skew_val = df[col].skew()
                if pd.notna(skew_val) and abs(skew_val) > self.limite_skew:
                    self.assimetricas_.append(col)
                else:
                    self.normais_.append(col)

        if self.verbose:
            print(
                f"  -> Prep (fit): {len(self.assimetricas_)} Yeo-Johnson | "
                f"{len(self.normais_)} StandardScaler | {len(self.oh_binarias_)} passthrough"
            )

        transformers = []
        if self.assimetricas_:
            transformers.append(
                ("yeo_johnson", PowerTransformer(method="yeo-johnson", standardize=True), self.assimetricas_)
            )
        if self.normais_:
            transformers.append(("padroniza", StandardScaler(), self.normais_))

        self.ct_ = ColumnTransformer(
            transformers=transformers,
            remainder="passthrough",
            verbose_feature_names_out=False,
        )
        if isinstance(X, pd.DataFrame):
            self.ct_.set_output(transform="pandas")

        self.ct_.fit(df, y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        return self.ct_.transform(df)

    def get_feature_names_out(self, input_features=None):
        return self.ct_.get_feature_names_out(input_features)


# ==========================================
# 2. Carregamento de Dados + Folds de Homologia
# ==========================================
def carregar_dados_e_cv(
    proc_dir: Path,
    n_splits: int = 5,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Any, Optional[np.ndarray]]:
    """
    Carrega X_train_limpo, X_test_limpo e prioriza meta_train.csv / meta_test.csv
    para preservar os folds de homologia 40% (cv_fold / cluster_id).
    """
    path_X_tr = proc_dir / "X_train_limpo.csv"
    path_X_te = proc_dir / "X_test_limpo.csv"
    for p in (path_X_tr, path_X_te):
        if not p.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {p}")

    X_tr = pd.read_csv(path_X_tr)
    X_te = pd.read_csv(path_X_te)

    path_meta_tr = proc_dir / "meta_train.csv"
    path_meta_te = proc_dir / "meta_test.csv"

    groups = None
    if path_meta_tr.exists() and path_meta_te.exists():
        meta_tr = pd.read_csv(path_meta_tr)
        meta_te = pd.read_csv(path_meta_te)
        y_tr = meta_tr["label"].astype(int)
        y_te = meta_te["label"].astype(int)

        if "cv_fold" in meta_tr.columns:
            cv = PredefinedSplit(test_fold=meta_tr["cv_fold"].to_numpy())
            print("  -> CV configurada via PredefinedSplit (meta_train['cv_fold'] - Homologia 40%).")
        elif "cluster_id" in meta_tr.columns:
            groups = meta_tr["cluster_id"].to_numpy()
            cv = StratifiedGroupKFold(n_splits=n_splits)
            print("  -> CV configurada via StratifiedGroupKFold (meta_train['cluster_id']).")
        else:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            print("  -> [AVISO] Sem cv_fold/cluster_id em meta_train. Usando StratifiedKFold padrão.")
    else:
        y_tr = pd.read_csv(proc_dir / "y_train.csv").squeeze("columns").astype(int)
        y_te = pd.read_csv(proc_dir / "y_test.csv").squeeze("columns").astype(int)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        print("  -> [AVISO] meta_train.csv não encontrado. Usando StratifiedKFold padrão.")

    return X_tr, X_te, y_tr, y_te, cv, groups


def calcular_scale_pos_weight(y: pd.Series) -> float:
    """Retorna razão neg/pos para balanceamento no XGBoost."""
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    return n_neg / max(n_pos, 1)


# ==========================================
# 3. Métricas e Scores
# ==========================================
def criar_scoring() -> Dict[str, Any]:
    return {
        "mcc": make_scorer(matthews_corrcoef),
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
        "sensibilidade": make_scorer(recall_score, pos_label=1),
        "especificidade": make_scorer(recall_score, pos_label=0),
        "acuracia": "accuracy",
    }


def pontuacao(modelo: Any, X: pd.DataFrame) -> np.ndarray:
    if hasattr(modelo, "predict_proba"):
        return modelo.predict_proba(X)[:, 1]
    if hasattr(modelo, "decision_function"):
        return modelo.decision_function(X)
    raise AttributeError("O modelo não possui 'predict_proba' nem 'decision_function'.")


def metricas_holdout(
    modelo: Any, X_te: pd.DataFrame, y_te: pd.Series
) -> Tuple[Dict[str, float], np.ndarray, Tuple[int, int, int, int]]:
    y_pred = modelo.predict(X_te)
    score = pontuacao(modelo, X_te)
    tn, fp, fn, tp = confusion_matrix(y_te, y_pred).ravel()

    brier = np.nan
    if hasattr(modelo, "predict_proba"):
        brier = brier_score_loss(y_te, score)

    metricas = {
        "mcc": matthews_corrcoef(y_te, y_pred),
        "roc_auc": roc_auc_score(y_te, score),
        "pr_auc": average_precision_score(y_te, score),
        "sensibilidade": tp / (tp + fn),
        "especificidade": tn / (tn + fp),
        "acuracia": accuracy_score(y_te, y_pred),
        "brier": brier,
    }
    return metricas, score, (tn, fp, fn, tp)


# ==========================================
# 4. Conformal Prediction (Incerteza Calibrada)
# ==========================================
def avaliar_conformal_prediction(
    modelo_base: Any,
    modelo_treinado: Any,
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_te: pd.DataFrame,
    y_te: pd.Series,
    cv: Any,
    groups: Optional[np.ndarray] = None,
    alpha: float = 0.10,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Conformal Prediction baseada em probabilidades out-of-fold (respeitando homologia 40%).
    Separa o teste cego em:
      - Alta confiança ABP ({1})
      - Alta confiança Não-ABP ({0})
      - Ambíguo ({0, 1}) — modelo assume que não sabe
      - Vazio ({}) — possível out-of-distribution
    """
    if not hasattr(modelo_treinado, "predict_proba"):
        raise ValueError("Conformal Prediction requer um modelo com predict_proba.")

    # 1. Probabilidades out-of-fold no treino (sem vazamento de homologia)
    probs_oof = cross_val_predict(
        clone(modelo_base), X_tr, y_tr, cv=cv, groups=groups, method="predict_proba", n_jobs=1
    )
    y_tr_arr = np.asarray(y_tr, dtype=int)
    scores_cal = 1.0 - probs_oof[np.arange(len(y_tr_arr)), y_tr_arr]

    # 2. Quantil conformal com correção de amostra finita
    n = len(scores_cal)
    nivel = min(np.ceil((n + 1) * (1.0 - alpha)) / n, 1.0)
    q_hat = float(np.quantile(scores_cal, nivel, method="higher"))

    # 3. Conjuntos de predição no teste cego
    probs_te = modelo_treinado.predict_proba(X_te)
    inclui_0 = (1.0 - probs_te[:, 0]) <= q_hat
    inclui_1 = (1.0 - probs_te[:, 1]) <= q_hat

    decisao = np.where(
        inclui_0 & inclui_1, "ambiguo",
        np.where(inclui_1, "abp_alta_confianca",
                 np.where(inclui_0, "nao_abp_alta_confianca", "vazio_ood"))
    )

    y_te_arr = np.asarray(y_te, dtype=int)
    coberto = np.where(y_te_arr == 1, inclui_1, inclui_0)

    mask_certeza = decisao != "ambiguo"
    mask_singleton = np.isin(decisao, ["abp_alta_confianca", "nao_abp_alta_confianca"])
    y_pred_sing = (decisao[mask_singleton] == "abp_alta_confianca").astype(int)
    mcc_singleton = (
        matthews_corrcoef(y_te_arr[mask_singleton], y_pred_sing)
        if mask_singleton.sum() > 0 and len(np.unique(y_te_arr[mask_singleton])) > 1
        else np.nan
    )

    resumo = {
        "alpha_alvo": alpha,
        "cobertura_alvo": 1.0 - alpha,
        "cobertura_empirica": float(coberto.mean()),
        "limiar_q_hat": round(q_hat, 4),
        "pct_alta_confianca": float(mask_singleton.mean() * 100),
        "pct_ambiguo": float((decisao == "ambiguo").mean() * 100),
        "pct_vazio_ood": float((decisao == "vazio_ood").mean() * 100),
        "mcc_alta_confianca": float(mcc_singleton),
    }

    df_pred = pd.DataFrame({
        "y_real": y_te_arr,
        "prob_abp": probs_te[:, 1],
        "inclui_0": inclui_0,
        "inclui_1": inclui_1,
        "decisao_conformal": decisao,
        "coberto": coberto,
    })
    return resumo, df_pred
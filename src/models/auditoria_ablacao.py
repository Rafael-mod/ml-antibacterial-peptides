"""
src/models/auditoria_ablacao.py

1. Quantifica viés de Metionina N-terminal (OHE_N1_M) e comprimento entre classes.
2. Executa estudo de ablação removendo grupos suspeitos nos mesmos folds (Homologia 40%).
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from sklearn.model_selection import PredefinedSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROC_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
TAB_DIR = ROOT / "reports" / "tabelas"


def carregar_pipeline_base():
    """Tenta clonar o pipeline interno do modelo campeão; senão usa SVM padrão."""
    caminho = MODELS_DIR / "modelo_final_otimizado.joblib"
    if caminho.exists():
        modelo = joblib.load(caminho)
        if hasattr(modelo, "estimator") and modelo.estimator is not None:
            return clone(modelo.estimator)
        if hasattr(modelo, "calibrated_classifiers_"):
            return clone(modelo.calibrated_classifiers_[0].estimator)
    return Pipeline([("scaler", StandardScaler()), ("clf", SVC(kernel="rbf", random_state=42))])


def diagnostico_vies(X_tr, y_tr, X_te, y_te):
    print("=== 1. DIAGNÓSTICO DE VIÉS NO DATASET ===")
    for nome, X, y in [("Treino", X_tr, y_tr), ("Teste Cego", X_te, y_te)]:
        print(f"\n[{nome}] (n={len(y)})")
        if "OHE_N1_M" in X.columns:
            p_pos = X.loc[y == 1, "OHE_N1_M"].mean() * 100
            p_neg = X.loc[y == 0, "OHE_N1_M"].mean() * 100
            print(f"  Começam com 'M' (OHE_N1_M=1) -> Positivos (ABP): {p_pos:.1f}% | Negativos: {p_neg:.1f}%")
        if "comprimento" in X.columns:
            for lbl, desc in [(1, "Positivos (ABP)"), (0, "Negativos      ")]:
                s = X.loc[y == lbl, "comprimento"]
                q1, med, q3 = np.percentile(s, [25, 50, 75])
                print(f"  Comprimento {desc} -> Mediana: {med:.1f} | IQR: [{q1:.1f}, {q3:.1f}] | Min-Max: [{s.min():.0f}, {s.max():.0f}]")


def rodar_ablacao():
    X_train = pd.read_csv(PROC_DIR / "X_train_limpo.csv")
    X_test = pd.read_csv(PROC_DIR / "X_test_limpo.csv")
    meta_train = pd.read_csv(PROC_DIR / "meta_train.csv")
    meta_test = pd.read_csv(PROC_DIR / "meta_test.csv")

    y_train = meta_train["label"].astype(int).values
    y_test = meta_test["label"].astype(int).values
    cv = PredefinedSplit(meta_train["cv_fold"].values)

    diagnostico_vies(X_train, y_train, X_test, y_test)

    ohe_n = [c for c in X_train.columns if c.startswith(("OHE_N", "OH_N"))]
    ohe_c = [c for c in X_train.columns if c.startswith(("OHE_C", "OH_C"))]

    cenarios = {
        "1. Completo (Baseline 81 feat.)": [],
        "2. Sem OHE_N1_M": ["OHE_N1_M"],
        "3. Sem OHE_N1_M + AAC_M (zero Metionina)": ["OHE_N1_M", "AAC_M"],
        "4. Sem terminais (remove OHE_N e OHE_C)": ohe_n + ohe_c,
        "5. Stress test (sem OHE_N/C, AAC_M e comprimento)": ohe_n + ohe_c + ["AAC_M", "comprimento"],
    }

    base_pipe = carregar_pipeline_base()
    resultados = []

    print("\n=== 2. ESTUDO DE ABLAÇÃO (SVM RBF — Folds Homologia 40%) ===")
    for nome, cols_remover in cenarios.items():
        cols_validas = [c for c in cols_remover if c in X_train.columns]
        X_tr_ab = X_train.drop(columns=cols_validas)
        X_te_ab = X_test.drop(columns=cols_validas)

        pipe = clone(base_pipe)
        scores_cv = cross_val_score(
            pipe, X_tr_ab, y_train, cv=cv, scoring="matthews_corrcoef", n_jobs=-1
        )

        pipe.fit(X_tr_ab, y_train)
        y_pred = pipe.predict(X_te_ab)
        score_te = (
            pipe.predict_proba(X_te_ab)[:, 1]
            if hasattr(pipe, "predict_proba")
            else pipe.decision_function(X_te_ab)
        )

        resultados.append({
            "cenario": nome,
            "n_features": X_tr_ab.shape[1],
            "mcc_cv_media": scores_cv.mean(),
            "mcc_cv_std": scores_cv.std(),
            "mcc_teste": matthews_corrcoef(y_test, y_pred),
            "roc_auc_teste": roc_auc_score(y_test, score_te),
        })

    df_res = pd.DataFrame(resultados)
    df_res["delta_mcc_teste"] = df_res["mcc_teste"] - df_res.loc[0, "mcc_teste"]

    print("\n" + df_res.round(4).to_string(index=False))
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    df_res.to_csv(TAB_DIR / "ablacao_vies_features.csv", index=False)
    print(f"\nTabela salva em: {(TAB_DIR / 'ablacao_vies_features.csv').relative_to(ROOT)}")


if __name__ == "__main__":
    rodar_ablacao()
"""
src/models/tuning.py (ou hiperparameters.py)

Busca de hiperparâmetros (RandomizedSearchCV) e exportação do modelo final.
Adaptado para a nova pipeline sem data leakage (carregando splits pré-definidos).
"""

import json
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import loguniform, randint, uniform

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             make_scorer, matthews_corrcoef, precision_recall_curve,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None
try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

sns.set_theme(style="whitegrid", context="notebook")

# ==========================================
# Constantes e Estrutura de Diretórios
# ==========================================
ROOT = Path(__file__).resolve().parents[2]
PROC_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"
PASTA_FIG = REPORTS_DIR / "figuras_tuning"
PASTA_TABELAS = REPORTS_DIR / "tabelas"
MODELS_DIR = ROOT / "models"

SEED = 42
N_JOBS_BUSCA = -1
LIMITE_SKEW = 1.0
MODELOS_ALVO = ("SVM (RBF)", "LightGBM", "XGBoost", "Random Forest")


# ==========================================
# 1. Pré-processamento
# ==========================================
def separar_grupos(df):
    aac = [c for c in df.columns if c.startswith("AAC_")]
    oh = [c for c in df.columns if c.startswith("OH_")]
    fis = [c for c in df.columns if c not in ("id", "sequence", "label", "classe") and c not in aac and c not in oh]
    return fis, aac, oh


def criar_preprocessador(X, limite=LIMITE_SKEW):
    fis, aac, _ = separar_grupos(X)
    
    fis_numericas = X[fis].select_dtypes(include=[np.number])
    skew = fis_numericas.skew()
    
    assimetricas = list(skew[skew.abs() > limite].index)
    normais = [c for c in fis_numericas.columns if c not in assimetricas]
    
    return ColumnTransformer(
        [("yeo_johnson", PowerTransformer(method="yeo-johnson", standardize=True), assimetricas),
         ("padroniza", StandardScaler(), normais + aac)],
        remainder="passthrough", verbose_feature_names_out=False,
    ).set_output(transform="pandas")


# ==========================================
# 2. Modelos base e espaços de busca
# ==========================================
def criar_estimadores(X_tr, n_arvores=300):
    est = {
        "SVM (RBF)": Pipeline([("prep", criar_preprocessador(X_tr)),
                               ("clf", SVC(kernel="rbf", random_state=SEED))]),
        "Random Forest": RandomForestClassifier(n_estimators=n_arvores, n_jobs=1, random_state=SEED),
    }
    if XGBClassifier is not None:
        est["XGBoost"] = XGBClassifier(n_estimators=n_arvores, learning_rate=0.1, max_depth=6,
                                       subsample=0.8, colsample_bytree=0.5, eval_metric="logloss",
                                       n_jobs=1, random_state=SEED)
    if LGBMClassifier is not None:
        est["LightGBM"] = LGBMClassifier(n_estimators=n_arvores, learning_rate=0.05, num_leaves=31,
                                         subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
                                         n_jobs=1, random_state=SEED, verbose=-1)
    return est


ESPACOS = {
    "SVM (RBF)": {
        "clf__C": loguniform(1e-1, 1e3),
        "clf__gamma": loguniform(1e-5, 1e-1),
    },
    "LightGBM": {
        "n_estimators": randint(100, 600),
        "learning_rate": loguniform(0.01, 0.2),
        "num_leaves": randint(15, 128),
        "max_depth": [-1, 4, 6, 8, 12],
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.2, 0.6),
        "min_child_samples": randint(5, 50),
        "reg_lambda": loguniform(1e-3, 10),
    },
    "XGBoost": {
        "n_estimators": randint(100, 600),
        "learning_rate": loguniform(0.01, 0.3),
        "max_depth": randint(3, 10),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.2, 0.6),
        "min_child_weight": randint(1, 8),
        "reg_lambda": loguniform(1e-3, 10),
    },
    "Random Forest": {
        "n_estimators": randint(100, 500),
        "max_depth": [None, 10, 20, 30],
        "max_features": ["sqrt", "log2", 0.1, 0.3],
        "min_samples_leaf": randint(1, 6),
    },
}


# ==========================================
# 3. Métricas
# ==========================================
def criar_scoring():
    return {"mcc": make_scorer(matthews_corrcoef), "roc_auc": "roc_auc", "pr_auc": "average_precision"}


def pontuacao(modelo, X):
    if hasattr(modelo, "predict_proba"):
        return modelo.predict_proba(X)[:, 1]
    return modelo.decision_function(X)


def metricas_holdout(modelo, X_te, y_te):
    y_pred = modelo.predict(X_te)
    score = pontuacao(modelo, X_te)
    tn, fp, fn, tp = confusion_matrix(y_te, y_pred).ravel()
    met = {"mcc": matthews_corrcoef(y_te, y_pred), "roc_auc": roc_auc_score(y_te, score),
           "pr_auc": average_precision_score(y_te, score), "sensibilidade": tp / (tp + fn),
           "especificidade": tn / (tn + fp), "acuracia": accuracy_score(y_te, y_pred)}
    return met, score


def _nativo(v):
    return v.item() if hasattr(v, "item") else v


# ==========================================
# 4. Busca de hiperparâmetros
# ==========================================
def otimizar(nome, base, X_tr, y_tr, cv, n_iter):
    scoring = criar_scoring()

    padrao = cross_validate(base, X_tr, y_tr, cv=cv, scoring=scoring, n_jobs=N_JOBS_BUSCA)

    t0 = time.time()
    busca = RandomizedSearchCV(base, ESPACOS[nome], n_iter=n_iter, scoring=scoring, refit="mcc",
                               cv=cv, n_jobs=N_JOBS_BUSCA, random_state=SEED, error_score="raise")
    busca.fit(X_tr, y_tr)
    i = busca.best_index_
    r = busca.cv_results_
    resumo = {
        "modelo": nome, "n_iter": n_iter, "tempo_busca_s": round(time.time() - t0, 1),
        "mcc_cv_padrao": padrao["test_mcc"].mean(), "mcc_cv_otimizado": r["mean_test_mcc"][i],
        "mcc_cv_otimizado_dp": r["std_test_mcc"][i],
        "roc_auc_cv_padrao": padrao["test_roc_auc"].mean(), "roc_auc_cv_otimizado": r["mean_test_roc_auc"][i],
        "pr_auc_cv_padrao": padrao["test_pr_auc"].mean(), "pr_auc_cv_otimizado": r["mean_test_pr_auc"][i],
    }
    params = {k: _nativo(v) for k, v in busca.best_params_.items()}
    print(f"  {nome:<14} MCC CV: padrão {resumo['mcc_cv_padrao']:.3f} -> otimizado "
          f"{resumo['mcc_cv_otimizado']:.3f} ± {resumo['mcc_cv_otimizado_dp']:.3f}  "
          f"({resumo['tempo_busca_s']}s)")
    print(f"     melhores parâmetros: {params}")
    return busca, resumo, params


# ==========================================
# 5. Gráficos
# ==========================================
def salvar(nome, fig):
    PASTA_FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(PASTA_FIG / nome, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_padrao_vs_otimizado(resumo):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (m, titulo) in zip(axes, [("mcc", "MCC"), ("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC")]):
        d = resumo.melt(id_vars="modelo", value_vars=[f"{m}_cv_padrao", f"{m}_cv_otimizado"],
                        var_name="config", value_name="valor")
        d["config"] = d["config"].map({f"{m}_cv_padrao": "Padrão", f"{m}_cv_otimizado": "Otimizado"})
        sns.barplot(data=d, x="modelo", y="valor", hue="config", ax=ax, palette=["#bdbdbd", "#DD8452"])
        lo = d["valor"].min()
        ax.set_ylim(max(0, lo - 0.05), min(1, d["valor"].max() + 0.02))
        ax.set_title(f"{titulo} na validação cruzada (treino)")
        ax.set_xlabel("")
        ax.legend(title="")
    salvar("01_padrao_vs_otimizado_cv.png", fig)


def plot_holdout(nome, y_te, score_pad, score_otm):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for rotulo, s, cor in [("Padrão", score_pad, "#7f7f7f"), ("Otimizado", score_otm, "#DD8452")]:
        fpr, tpr, _ = roc_curve(y_te, s)
        axes[0].plot(fpr, tpr, lw=2, color=cor, label=f"{rotulo} (AUC = {roc_auc_score(y_te, s):.3f})")
        p, r, _ = precision_recall_curve(y_te, s)
        axes[1].plot(r, p, lw=2, color=cor, label=f"{rotulo} (PR-AUC = {average_precision_score(y_te, s):.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[0].set(title="ROC (teste cego)", xlabel="1 − Especificidade", ylabel="Sensibilidade")
    axes[1].set(title="Precisão-Recall (teste cego)", xlabel="Recall", ylabel="Precisão", ylim=(0, 1.02))
    for ax in axes:
        ax.legend(loc="lower right" if ax is axes[0] else "lower left")
    fig.suptitle(f"{nome}: padrão vs otimizado", y=1.02, fontsize=14)
    salvar("02_holdout_padrao_vs_otimizado.png", fig)


# ==========================================
# 6. Execução Principal
# ==========================================
def main(n_splits=5, n_iter=60):
    
    # 1) Carregamento Direto dos Dados Limpos
    path_X_tr = PROC_DIR / "X_train_limpo.csv"
    path_X_te = PROC_DIR / "X_test_limpo.csv"
    path_y_tr = PROC_DIR / "y_train.csv"
    path_y_te = PROC_DIR / "y_test.csv"
    
    for p in [path_X_tr, path_X_te, path_y_tr, path_y_te]:
        if not p.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {p}")

    X_tr = pd.read_csv(path_X_tr)
    X_te = pd.read_csv(path_X_te)
    y_tr = pd.read_csv(path_y_tr).squeeze("columns")
    y_te = pd.read_csv(path_y_te).squeeze("columns")

    print(f"Treino: {len(X_tr)} amostras | Teste cego: {len(X_te)} amostras")

    base = criar_estimadores(X_tr)
    alvo = [m for m in MODELOS_ALVO if m in base]

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    print(f"\nRandomizedSearchCV ({n_iter} combinações x {n_splits} dobras, refit por MCC):")
    
    buscas, linhas, melhores = {}, [], {}
    for nome in alvo:
        busca, resumo, params = otimizar(nome, base[nome], X_tr, y_tr, cv, n_iter)
        buscas[nome], melhores[nome] = busca, params
        linhas.append(resumo)
        
    PASTA_TABELAS.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    resumo = pd.DataFrame(linhas).sort_values("mcc_cv_otimizado", ascending=False).reset_index(drop=True)
    resumo.to_csv(PASTA_TABELAS / "resultados_tuning_cv.csv", index=False)
    
    with open(MODELS_DIR / "melhores_hiperparametros.json", "w", encoding="utf-8") as f:
        json.dump(melhores, f, indent=2, ensure_ascii=False)
        
    plot_padrao_vs_otimizado(resumo)

    # Avaliação Final no Teste Cego
    campeao = resumo.loc[0, "modelo"]
    otimizado = buscas[campeao].best_estimator_ 
    padrao = clone(base[campeao]).fit(X_tr, y_tr)
    
    met_pad, s_pad = metricas_holdout(padrao, X_te, y_te)
    met_otm, s_otm = metricas_holdout(otimizado, X_te, y_te)
    
    teste = pd.DataFrame([{"modelo": campeao, "config": "padrão", **met_pad},
                          {"modelo": campeao, "config": "otimizado", **met_otm}])
    teste.to_csv(PASTA_TABELAS / "resultados_teste_cego_tuning.csv", index=False)
    
    print(f"\nCampeão da CV: {campeao}. Desempenho no teste cego:")
    print(teste.round(3).to_string(index=False))
    plot_holdout(campeao, y_te, s_pad, s_otm)

    # Guarda o modelo treinado
    caminho_modelo = MODELS_DIR / "modelo_final_otimizado.joblib"
    joblib.dump(otimizado, caminho_modelo)
    print(f"\nModelo otimizado salvo em: {caminho_modelo.relative_to(ROOT)}")
    
    return resumo, teste


if __name__ == "__main__":
    main()
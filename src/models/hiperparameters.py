"""
src/models/tuning.py

Busca de hiperparâmetros (RandomizedSearchCV) e exportação do modelo final.
Foco em execução rápida (Hist-trees, SVM Cache) e prevenção de over-tuning.
"""

import json
import time
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import loguniform, randint

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             make_scorer, matthews_corrcoef, precision_recall_curve,
                             roc_auc_score, roc_curve)
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
# 1. Pré-processamento Genérico e Inteligente
# ==========================================
def criar_preprocessador(X: pd.DataFrame, limite_skew: float = LIMITE_SKEW) -> ColumnTransformer:
    oh_binarias = [c for c in X.columns if c.startswith("OH_") or X[c].nunique() <= 2]
    resto = [c for c in X.columns if c not in oh_binarias]
    
    assimetricas, normais = [], []
    for col in resto:
        if (X[col] == 0).mean() > 0.30 or abs(X[col].skew()) <= limite_skew:
            normais.append(col)
        else:
            assimetricas.append(col)
                
    return ColumnTransformer(
        transformers=[
            ("yeo_johnson", PowerTransformer(method="yeo-johnson", standardize=True), assimetricas),
            ("padroniza", StandardScaler(), normais)
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")


# ==========================================
# 2. Modelos base e espaços de busca
# ==========================================
def criar_estimadores(X_tr):
    # Otimização 1: cache_size=2000 acelera o SVM RBF usando mais RAM
    est = {
        "SVM (RBF)": Pipeline([
            ("prep", criar_preprocessador(X_tr)),
            ("clf", SVC(kernel="rbf", random_state=SEED, cache_size=2000)) 
        ]),
        "Random Forest": RandomForestClassifier(n_jobs=1, random_state=SEED),
    }
    
    # Otimização 2: tree_method="hist" acelera o XGBoost massivamente
    if XGBClassifier is not None:
        est["XGBoost"] = XGBClassifier(eval_metric="logloss", n_jobs=1, random_state=SEED, 
                                       tree_method="hist")
    if LGBMClassifier is not None:
        est["LightGBM"] = LGBMClassifier(n_jobs=1, random_state=SEED, verbose=-1)
        
    return est


# Grids ajustados: Menos agressivos no número de estimadores para acelerar a busca
ESPACOS = {
    "SVM (RBF)": {
        "clf__C": loguniform(0.5, 20.0), # Grid mais estreito para evitar overfitting
        "clf__gamma": loguniform(0.001, 0.05),
    },
    "LightGBM": {
        "max_depth": [4, 5, 6],
        "num_leaves": [15, 31, 45],
        "min_child_samples": [30, 50, 70],
        "colsample_bytree": [0.4, 0.6, 0.8],
        "subsample": [0.7, 0.8, 0.9],
        "n_estimators": [100, 150, 200],
        "learning_rate": [0.03, 0.05, 0.1],
        "reg_alpha": [0.1, 1.0, 5.0],
        "reg_lambda": [1.0, 5.0, 10.0],
    },
    "XGBoost": {
        "max_depth": [4, 5, 6],
        "min_child_weight": [3, 5, 7],
        "colsample_bytree": [0.4, 0.6, 0.8],
        "subsample": [0.7, 0.8, 0.9],
        "n_estimators": [100, 150, 200],
        "learning_rate": [0.03, 0.05, 0.1],
        "reg_alpha": [0.1, 1.0, 5.0],
        "reg_lambda": [1.0, 5.0, 10.0],
    },
    "Random Forest": {
        "n_estimators": randint(100, 250), # Teto reduzido de 500 para 250 (Ganha muita velocidade)
        "max_depth": [10, 15, None],
        "max_features": ["sqrt", "log2", 0.3],
        "min_samples_leaf": randint(2, 6),
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
    return {
        "mcc": matthews_corrcoef(y_te, y_pred), "roc_auc": roc_auc_score(y_te, score),
        "pr_auc": average_precision_score(y_te, score), "sensibilidade": tp / (tp + fn),
        "especificidade": tn / (tn + fp), "acuracia": accuracy_score(y_te, y_pred)
    }, score

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
        ax.set_ylim(max(0, d["valor"].min() - 0.05), min(1, d["valor"].max() + 0.02))
        ax.set(title=f"{titulo} (treino)", xlabel="")
        ax.legend(title="")
    salvar("01_padrao_vs_otimizado_cv.png", fig)

def plot_holdout(nome, y_te, score_pad, score_otm, filename):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for rotulo, s, cor in [("Padrão", score_pad, "#7f7f7f"), ("Otimizado", score_otm, "#DD8452")]:
        fpr, tpr, _ = roc_curve(y_te, s)
        axes[0].plot(fpr, tpr, lw=2, color=cor, label=f"{rotulo} (AUC = {roc_auc_score(y_te, s):.3f})")
        p, r, _ = precision_recall_curve(y_te, s)
        axes[1].plot(r, p, lw=2, color=cor, label=f"{rotulo} (PR-AUC = {average_precision_score(y_te, s):.3f})")
    
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[0].set(title="ROC (teste cego)", xlabel="1 − Especificidade", ylabel="Sensibilidade")
    axes[1].set(title="Precisão-Recall", xlabel="Recall", ylabel="Precisão", ylim=(0, 1.02))
    for ax in axes: ax.legend(loc="lower right" if ax is axes[0] else "lower left")
    fig.suptitle(f"{nome}: padrão vs otimizado", y=1.02, fontsize=14)
    salvar(filename, fig)


# ==========================================
# 6. Execução Principal
# ==========================================
def main(n_splits=5, n_iter=35): # Otimização 4: n_iter reduzido de 60 para 35
    
    path_X_tr = PROC_DIR / "X_train_limpo.csv"
    path_X_te = PROC_DIR / "X_test_limpo.csv"
    path_y_tr = PROC_DIR / "y_train.csv"
    path_y_te = PROC_DIR / "y_test.csv"
    
    X_tr = pd.read_csv(path_X_tr)
    X_te = pd.read_csv(path_X_te)
    y_tr = pd.read_csv(path_y_tr).squeeze("columns")
    y_te = pd.read_csv(path_y_te).squeeze("columns")

    print(f"Treino: {len(X_tr)} amostras | Teste cego: {len(X_te)} amostras")

    base = criar_estimadores(X_tr)
    alvo = [m for m in MODELOS_ALVO if m in base]
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    
    print(f"\nRandomizedSearchCV ({n_iter} comb. x {n_splits} dobras):")
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

    print("\nCalculando métricas no Teste Cego...")
    linhas_teste = []
    
    for nome in alvo:
        otimizado = buscas[nome].best_estimator_ 
        padrao = clone(base[nome]).fit(X_tr, y_tr)
        
        met_pad, s_pad = metricas_holdout(padrao, X_te, y_te)
        met_otm, s_otm = metricas_holdout(otimizado, X_te, y_te)
        
        linhas_teste.extend([
            {"modelo": nome, "config": "padrão", **met_pad},
            {"modelo": nome, "config": "otimizado", **met_otm}
        ])
        plot_holdout(nome, y_te, s_pad, s_otm, f"02_holdout_{nome.replace(' ', '_')}.png")
    
    teste = pd.DataFrame(linhas_teste)
    teste.to_csv(PASTA_TABELAS / "resultados_teste_cego.csv", index=False)
    print("\nDesempenho no Teste Cego:")
    print(teste.round(3).to_string(index=False))

    campeao = resumo.loc[0, "modelo"]
    modelo_final = buscas[campeao].best_estimator_
    caminho = MODELS_DIR / "modelo_final_otimizado.joblib"
    joblib.dump(modelo_final, caminho)
    print(f"\n[!] Campeão ({campeao}) salvo em: {caminho.relative_to(ROOT)}")
    
    return resumo, teste

if __name__ == "__main__":
    main()
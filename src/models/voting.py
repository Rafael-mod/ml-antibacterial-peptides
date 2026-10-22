"""
src/models/voting.py

Módulo de Treinamento de Ensembles (Voting e Stacking):
1. Carrega os melhores hiperparâmetros do tuning.py.
2. Configura os estimadores base: LightGBM, XGBoost e SVM (Calibrado).
3. Constrói o Soft Voting e o Stacking Classifier.
4. Aplica validação cruzada e treina no conjunto completo.
5. Exporta métricas comparativas, gráficos e os modelos finais.
"""

import json
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Any, Dict, List, Tuple

from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, make_scorer, matthews_corrcoef,
                             precision_recall_curve, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.svm import SVC

# Imports opcionais com graceful fallback
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

# Configuração de Estilo Global
sns.set_theme(style="whitegrid", context="notebook")

# ==========================================
# Constantes do Projeto
# ==========================================
ROOT = Path(__file__).resolve().parents[2]
PROC_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"
PASTA_FIG = REPORTS_DIR / "figuras_ensembles"
PASTA_TABELAS = REPORTS_DIR / "tabelas"
MODELS_DIR = ROOT / "models"

SEED = 42
N_JOBS = -1
LIMITE_SKEW = 1.0

# ==========================================
# 1. Pré-processamento
# ==========================================
def separar_grupos(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    aac = [c for c in df.columns if c.startswith("AAC_")]
    oh = [c for c in df.columns if c.startswith("OH_")]
    fis = [c for c in df.columns if c not in ("sequence", "label", "classe") and c not in aac and c not in oh]
    return fis, aac, oh


def criar_preprocessador(X: pd.DataFrame, limite: float = LIMITE_SKEW) -> ColumnTransformer:
    fis, aac, _ = separar_grupos(X)
    
    fis_numericas = X[fis].select_dtypes(include=[np.number])
    skew = fis_numericas.skew()
    
    assimetricas = list(skew[skew.abs() > limite].index)
    normais = [c for c in fis_numericas.columns if c not in assimetricas]
    
    return ColumnTransformer(
        transformers=[
            ("yeo_johnson", PowerTransformer(method="yeo-johnson", standardize=True), assimetricas),
            ("padroniza", StandardScaler(), normais + aac)
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")


# ==========================================
# 2. Configuração dos Modelos Base
# ==========================================
def obter_estimadores_base(X_tr: pd.DataFrame) -> List[Tuple[str, Any]]:
    caminho_params = MODELS_DIR / "melhores_hiperparametros.json"
    if not caminho_params.exists():
        raise FileNotFoundError(f"Execute o tuning.py primeiro! Ficheiro não encontrado: {caminho_params}")
        
    with open(caminho_params, "r", encoding="utf-8") as f:
        best_params = json.load(f)
        
    estimadores = []

    # SVM (RBF)
    svc_base = SVC(kernel="rbf", random_state=SEED)
    svc_calibrado = CalibratedClassifierCV(estimator=svc_base, ensemble=False)

    svm_pipe = Pipeline([
        ("prep", criar_preprocessador(X_tr)),
        ("clf", svc_calibrado)
    ])

    params_svm = best_params.get("SVM (RBF)", {})
    params_ajustados = {}
    for k, v in params_svm.items():
        if k.startswith("clf__"):
            params_ajustados[k.replace("clf__", "clf__estimator__")] = v
        else:
            params_ajustados[f"clf__estimator__{k}"] = v

    svm_pipe.set_params(**params_ajustados)
    estimadores.append(("SVM", svm_pipe))

    # LightGBM
    if LGBMClassifier is not None:
        lgbm = LGBMClassifier(random_state=SEED, n_jobs=1, verbose=-1)
        lgbm.set_params(**best_params.get("LightGBM", {}))
        estimadores.append(("LightGBM", lgbm))

    # XGBoost
    if XGBClassifier is not None:
        xgb = XGBClassifier(random_state=SEED, eval_metric="logloss", n_jobs=1)
        xgb.set_params(**best_params.get("XGBoost", {}))
        estimadores.append(("XGBoost", xgb))

    return estimadores

def criar_voting_classifier(estimadores: List[Tuple[str, Any]]) -> VotingClassifier:
    return VotingClassifier(estimators=estimadores, voting="soft", weights=[2.34, 2.50, 2.12], n_jobs=N_JOBS)

def criar_stacking_classifier(estimadores: List[Tuple[str, Any]]) -> StackingClassifier:
    return StackingClassifier(
        estimators=estimadores,
        final_estimator=LogisticRegression(random_state=SEED, class_weight="balanced"),
        cv=5,
        n_jobs=N_JOBS
    )


# ==========================================
# 3. Métricas e Avaliação
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
    return modelo.predict_proba(X)[:, 1]

def metricas_holdout(modelo: Any, X_te: pd.DataFrame, y_te: pd.Series) -> Tuple[Dict[str, float], np.ndarray, Tuple[int, int, int, int]]:
    y_pred = modelo.predict(X_te)
    score = pontuacao(modelo, X_te)
    tn, fp, fn, tp = confusion_matrix(y_te, y_pred).ravel()
    
    metricas = {
        "mcc": matthews_corrcoef(y_te, y_pred),
        "roc_auc": roc_auc_score(y_te, score),
        "pr_auc": average_precision_score(y_te, score),
        "sensibilidade": tp / (tp + fn),
        "especificidade": tn / (tn + fp),
        "acuracia": accuracy_score(y_te, y_pred),
    }
    return metricas, score, (tn, fp, fn, tp)


# ==========================================
# 4. Exportação Visual (Gráficos)
# ==========================================
def salvar(nome: str, fig: plt.Figure) -> None:
    PASTA_FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(PASTA_FIG / nome, dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_cv(res_cv: dict, nome_modelo: str) -> None:
    metricas = [("test_mcc", "MCC"), ("test_roc_auc", "ROC-AUC"), ("test_pr_auc", "PR-AUC")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    
    for ax, (m, titulo) in zip(axes, metricas):
        dados = pd.DataFrame({nome_modelo: res_cv[m]})
        sns.boxplot(data=dados, orient="h", ax=ax, color="#9ecae1", fliersize=0)
        sns.stripplot(data=dados, orient="h", ax=ax, color="k", size=6, alpha=0.7)
        ax.set_title(f"{titulo} por Dobra (5-fold CV)")
        
    fig.suptitle(f"Validação Cruzada: {nome_modelo}", y=1.05, fontsize=14, fontweight="bold")
    fig.tight_layout()
    salvar(f"01_cv_{nome_modelo.lower().replace(' ', '_')}.png", fig)

def plot_holdout(y_te: pd.Series, score: np.ndarray, matriz: Tuple[int, int, int, int], nome_modelo: str) -> None:
    tn, fp, fn, tp = matriz
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    
    fpr, tpr, _ = roc_curve(y_te, score)
    axes[0].plot(fpr, tpr, lw=2, color="#d62728", label=f"AUC = {roc_auc_score(y_te, score):.3f}")
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[0].set(title="Curva ROC", xlabel="1 − Especificidade", ylabel="Sensibilidade")
    axes[0].legend(loc="lower right")

    prec, rec, _ = precision_recall_curve(y_te, score)
    axes[1].plot(rec, prec, lw=2, color="#1f77b4", label=f"PR-AUC = {average_precision_score(y_te, score):.3f}")
    axes[1].set(title="Curva Precisão-Recall", xlabel="Recall", ylabel="Precisão", ylim=(0, 1.02))
    axes[1].legend(loc="lower left")

    sns.heatmap(np.array([[tn, fp], [fn, tp]]), annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Prev. Neg", "Prev. Pos"], yticklabels=["Real Neg", "Real Pos"], ax=axes[2])
    axes[2].set_title("Matriz de Confusão")
    
    fig.suptitle(f"Desempenho Final (Teste Cego): {nome_modelo}", y=1.05, fontsize=14, fontweight="bold")
    salvar(f"02_holdout_{nome_modelo.lower().replace(' ', '_')}.png", fig)


# ==========================================
# 5. Pipeline Principal de Execução
# ==========================================
def main() -> None:
    caminho = PROC_DIR / "dataset_treino_limpo.csv"
    
    print("=== Iniciando Treino de Ensembles (Voting e Stacking) ===")
    if not caminho.exists():
        raise FileNotFoundError(f"Ficheiro não encontrado: {caminho}")

    df = pd.read_csv(caminho)
    y = df["label"]
    X = df.drop(columns=["sequence", "label", "classe"], errors="ignore").select_dtypes(include=[np.number])

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    print(f"Treino: {len(X_tr)} amostras | Teste Cego: {len(X_te)} amostras\n")

    estimadores_base = obter_estimadores_base(X_tr)
    
    modelos = {
        "Voting Soft": criar_voting_classifier(estimadores_base),
        "Stacking": criar_stacking_classifier(estimadores_base)
    }

    resultados_cv = []
    resultados_teste = []
    
    PASTA_TABELAS.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    for nome, clf in modelos.items():
        print(f"--- Avaliando: {nome} ---")
        
        # 1. Validação Cruzada
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        scoring = criar_scoring()
        
        t0 = time.time()
        res_cv = cross_validate(clf, X_tr, y_tr, cv=cv, scoring=scoring, n_jobs=1)
        tempo = round(time.time() - t0, 1)
        
        linha_cv = {"modelo": nome, "tempo_s": tempo}
        for m in scoring:
            linha_cv[f"{m}_media"] = res_cv[f"test_{m}"].mean()
            linha_cv[f"{m}_dp"] = res_cv[f"test_{m}"].std()
            
        resultados_cv.append(linha_cv)
        plot_cv(res_cv, nome)
        
        print(f"  CV -> MCC: {linha_cv['mcc_media']:.3f} ± {linha_cv['mcc_dp']:.3f} | AUC: {linha_cv['roc_auc_media']:.3f} ({tempo}s)")

        # 2. Teste Cego
        clf.fit(X_tr, y_tr)
        met, score, matriz = metricas_holdout(clf, X_te, y_te)
        resultados_teste.append({"modelo": nome, **met})
        plot_holdout(y_te, score, matriz, nome)
        
        # 3. Guardar Modelo
        caminho_modelo = MODELS_DIR / f"{nome.lower().replace(' ', '_')}_final.joblib"
        joblib.dump(clf, caminho_modelo)
        print(f"  Modelo salvo em: {caminho_modelo.relative_to(ROOT)}\n")

    # Exportar Tabelas Comparativas
    df_cv_final = pd.DataFrame(resultados_cv)
    df_teste_final = pd.DataFrame(resultados_teste)
    
    df_cv_final.to_csv(PASTA_TABELAS / "resultados_cv_ensembles.csv", index=False)
    df_teste_final.to_csv(PASTA_TABELAS / "resultados_teste_cego_ensembles.csv", index=False)

    print("=== Comparativo no Teste Cego ===")
    print(df_teste_final.round(3).to_string(index=False))
    print("\nProcesso concluído com sucesso!")


if __name__ == "__main__":
    main()
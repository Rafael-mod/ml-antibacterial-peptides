"""
src/models/treinamento.py

Módulo de Treinamento e Avaliação de Modelos:
1. Carrega os dados limpos já separados em Treino e Teste Cego.
2. Aplica validação cruzada (5-fold) estratificada exclusivamente no conjunto de treino.
3. Treina o melhor modelo e avalia o seu desempenho no teste cego.
4. Gera e guarda gráficos de desempenho e exporta o modelo final treinado.
"""

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, make_scorer, matthews_corrcoef,
                             precision_recall_curve, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_validate
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
MODELS_DIR = ROOT / "models"
PASTA_FIG = REPORTS_DIR / "figuras_modelagem"
PASTA_TABELAS = REPORTS_DIR / "tabelas"

SEED = 42
N_JOBS = -1
LIMITE_SKEW = 1.0

# ==========================================
# 1. Pré-processamento
# ==========================================
def separar_grupos(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    """Separa as features numéricas clássicas, PseAAC/Latentes, AAC e One-Hot."""
    aac = [c for c in df.columns if c.startswith("AAC_")]
    oh = [c for c in df.columns if c.startswith("OH_")]
    fis = [c for c in df.columns if c not in ("id", "sequence", "label", "classe") and c not in aac and c not in oh]
    return fis, aac, oh


def criar_preprocessador(X: pd.DataFrame, limite: float = LIMITE_SKEW) -> ColumnTransformer:
    """
    Aplica transformação de Yeo-Johnson em colunas assimétricas e StandardScaler nas contínuas.
    Mantém as representações One-Hot intactas. Ajustado dinamicamente com base no Treino.
    """
    fis, aac, _ = separar_grupos(X)
    
    fis_numericas = X[fis].select_dtypes(include=[np.number])
    skew = fis_numericas.skew()
    
    assimetricas = list(skew[skew.abs() > limite].index)
    normais = [c for c in fis_numericas.columns if c not in assimetricas]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("yeo_johnson", PowerTransformer(method="yeo-johnson", standardize=True), assimetricas),
            ("padroniza", StandardScaler(), normais + aac)
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    return preprocessor.set_output(transform="pandas")


# ==========================================
# 2. Configuração de Modelos
# ==========================================
def criar_modelos(X_tr: pd.DataFrame, n_arvores: int = 300) -> Dict[str, Any]:
    """Instancia o dicionário de modelos, injetando pipelines onde o scaling é necessário."""
    svc_base = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=SEED)
    svc_calibrado = CalibratedClassifierCV(estimator=svc_base, ensemble=False)

    modelos = {
        "Regressão Logística": Pipeline([
            ("prep", criar_preprocessador(X_tr)),
            ("clf", LogisticRegression(max_iter=3000, random_state=SEED))
        ]),
        "SVM (RBF)": Pipeline([
            ("prep", criar_preprocessador(X_tr)),
            ("clf", svc_calibrado)
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=n_arvores, n_jobs=N_JOBS, random_state=SEED
        ),
    }

    if XGBClassifier is not None:
        modelos["XGBoost"] = XGBClassifier(
            n_estimators=n_arvores, learning_rate=0.1, max_depth=6, subsample=0.8,
            colsample_bytree=0.5, eval_metric="logloss", n_jobs=N_JOBS, random_state=SEED
        )
    else:
        print("  -> Aviso: XGBoost não instalado. Modelo ignorado.")

    if LGBMClassifier is not None:
        modelos["LightGBM"] = LGBMClassifier(
            n_estimators=n_arvores, learning_rate=0.05, num_leaves=31, subsample=0.8,
            subsample_freq=1, colsample_bytree=0.5, n_jobs=N_JOBS, random_state=SEED, verbose=-1
        )
    else:
        print("  -> Aviso: LightGBM não instalado. Modelo ignorado.")

    return modelos


# ==========================================
# 3. Métricas e Avaliação
# ==========================================
def criar_scoring() -> Dict[str, Any]:
    """Define o conjunto de métricas de avaliação do Scikit-Learn."""
    return {
        "mcc": make_scorer(matthews_corrcoef),
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
        "sensibilidade": make_scorer(recall_score, pos_label=1),
        "especificidade": make_scorer(recall_score, pos_label=0),
        "acuracia": "accuracy",
    }


def pontuacao(modelo: Any, X: pd.DataFrame) -> np.ndarray:
    """Extrai scores contínuos para cálculo de curvas ROC e PR."""
    if hasattr(modelo, "predict_proba"):
        try:
            return modelo.predict_proba(X)[:, 1]
        except AttributeError:
            pass
    if hasattr(modelo, "decision_function"):
        return modelo.decision_function(X)
    raise AttributeError("O modelo não possui os métodos 'predict_proba' ou 'decision_function'.")


def metricas_holdout(modelo: Any, X_te: pd.DataFrame, y_te: pd.Series) -> Tuple[Dict[str, float], np.ndarray, Tuple[int, int, int, int]]:
    """Calcula as métricas definitivas no conjunto de teste cego."""
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


def validacao_cruzada(modelos: Dict[str, Any], X_tr: pd.DataFrame, y_tr: pd.Series, n_splits: int = 5) -> Tuple[pd.DataFrame, Dict[str, Dict[str, np.ndarray]]]:
    """Executa k-fold cross-validation estratificada e consolida os resultados."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    scoring = criar_scoring()
    resumo, por_dobra = [], {}
    
    for nome, modelo in modelos.items():
        t0 = time.time()
        res = cross_validate(modelo, X_tr, y_tr, cv=cv, scoring=scoring, n_jobs=1)
        tempo = round(time.time() - t0, 1)
        
        linha = {"modelo": nome, "tempo_s": tempo}
        for m in scoring:
            val = res[f"test_{m}"]
            linha[f"{m}_media"], linha[f"{m}_dp"] = val.mean(), val.std()
        
        resumo.append(linha)
        por_dobra[nome] = {m: res[f"test_{m}"] for m in scoring}
        
        print(f"  -> {nome:<20} MCC: {linha['mcc_media']:.3f} ± {linha['mcc_dp']:.3f} | AUC: {linha['roc_auc_media']:.3f} ({tempo}s)")
        
    df_resumo = pd.DataFrame(resumo).sort_values("mcc_media", ascending=False).reset_index(drop=True)
    return df_resumo, por_dobra


def tabela_formatada(resumo: pd.DataFrame) -> pd.DataFrame:
    """Gera uma string de representação amigável dos desvios padrões (Ex: '0.850 ± 0.021')."""
    t = pd.DataFrame({"modelo": resumo["modelo"]})
    for m in ["mcc", "roc_auc", "pr_auc", "sensibilidade", "especificidade", "acuracia"]:
        t[m] = resumo[f"{m}_media"].map("{:.3f}".format) + " ± " + resumo[f"{m}_dp"].map("{:.3f}".format)
    return t


# ==========================================
# 4. Exportação Visual (Gráficos)
# ==========================================
def salvar(nome: str, fig: plt.Figure) -> None:
    """Guarda a figura na diretoria definida sem bloquear a execução do script."""
    PASTA_FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(PASTA_FIG / nome, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_cv(por_dobra: Dict[str, Dict[str, np.ndarray]], ordem: List[str]) -> None:
    metricas = [("mcc", "MCC"), ("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    
    for ax, (m, titulo) in zip(axes, metricas):
        dados = pd.DataFrame({n: por_dobra[n][m] for n in ordem})
        sns.boxplot(data=dados, orient="h", ax=ax, color="#9ecae1", fliersize=0)
        sns.stripplot(data=dados, orient="h", ax=ax, color="k", size=4, alpha=0.7)
        ax.set_title(f"{titulo} por Dobra (5-fold CV)")
        
    fig.tight_layout()
    salvar("01_cv_benchmark.png", fig)


def plot_holdout(nome: str, y_te: pd.Series, score: np.ndarray, matriz: Tuple[int, int, int, int]) -> None:
    tn, fp, fn, tp = matriz
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_te, score)
    axes[0].plot(fpr, tpr, lw=2, color="#d62728", label=f"AUC = {roc_auc_score(y_te, score):.3f}")
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[0].set(title="Curva ROC (Teste Cego)", xlabel="1 − Especificidade", ylabel="Sensibilidade")
    axes[0].legend(loc="lower right")

    # PR Curve
    prec, rec, _ = precision_recall_curve(y_te, score)
    axes[1].plot(rec, prec, lw=2, color="#1f77b4", label=f"PR-AUC = {average_precision_score(y_te, score):.3f}")
    axes[1].set(title="Curva Precisão-Recall (Teste Cego)", xlabel="Recall", ylabel="Precisão", ylim=(0, 1.02))
    axes[1].legend(loc="lower left")

    # Matriz de Confusão
    sns.heatmap(np.array([[tn, fp], [fn, tp]]), annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Prev. Neg", "Prev. Pos"], yticklabels=["Real Neg", "Real Pos"], ax=axes[2])
    axes[2].set_title("Matriz de Confusão (Teste Cego)")
    
    fig.suptitle(f"Desempenho do Modelo Selecionado: {nome}", y=1.05, fontsize=14, fontweight="bold")
    salvar("02_holdout_melhor_modelo.png", fig)


# ==========================================
# 5. Pipeline Principal de Execução
# ==========================================
def main(
    path_X_tr: Path = PROC_DIR / "X_train_limpo.csv",
    path_X_te: Path = PROC_DIR / "X_test_limpo.csv",
    path_y_tr: Path = PROC_DIR / "y_train.csv",
    path_y_te: Path = PROC_DIR / "y_test.csv",
    n_splits: int = 5, 
    n_arvores: int = 300, 
    avaliar_todos_no_teste: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    
    print("=== Iniciando Pipeline de Modelação ===")
    
    # 1) Carregamento Direto dos Dados Limpos
    for p in [path_X_tr, path_X_te, path_y_tr, path_y_te]:
        if not p.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {p}")

    X_tr = pd.read_csv(path_X_tr)
    X_te = pd.read_csv(path_X_te)
    # Squeeze converte os targets de DataFrame para Series unidimensionais
    y_tr = pd.read_csv(path_y_tr).squeeze("columns")
    y_te = pd.read_csv(path_y_te).squeeze("columns")
    
    print(f"Conjunto de Treino: {X_tr.shape[0]} amostras, {X_tr.shape[1]} features")
    print(f"Conjunto de Teste Cego: {X_te.shape[0]} amostras, {X_te.shape[1]} features")

    # 2) Definição e Avaliação via CV (Apenas no Treino)
    modelos = criar_modelos(X_tr, n_arvores)
    print(f"\n[1/3] Validação Cruzada Estratificada ({n_splits}-fold) no Treino:")
    resumo, por_dobra = validacao_cruzada(modelos, X_tr, y_tr, n_splits)
    
    PASTA_TABELAS.mkdir(parents=True, exist_ok=True)
    resumo.to_csv(PASTA_TABELAS / "resultados_cv.csv", index=False)
    
    print("\nResumo da CV (Média ± Desvio Padrão):")
    print(tabela_formatada(resumo).to_string(index=False))
    plot_cv(por_dobra, list(resumo["modelo"]))

    # 3) Avaliação no Teste Cego e Exportação
    candidatos = list(resumo["modelo"]) if avaliar_todos_no_teste else [resumo.loc[0, "modelo"]]
    print(f"\n[2/3] Avaliando modelo(s) no Teste Cego: {', '.join(candidatos)}")
    
    linhas, melhor_resultado = [], None
    for nome in candidatos:
        modelo_treinado = clone(modelos[nome]).fit(X_tr, y_tr)
        met, score, matriz = metricas_holdout(modelo_treinado, X_te, y_te)
        linhas.append({"modelo": nome, **met})
        
        if melhor_resultado is None:
            melhor_resultado = (nome, score, matriz, modelo_treinado)
            
    teste = pd.DataFrame(linhas)
    teste.to_csv(PASTA_TABELAS / "resultados_teste_cego.csv", index=False)
    
    print("\nDesempenho Final (Holdout Cego):")
    print(teste.round(3).to_string(index=False))
    
    nome_melhor, score_melhor, matriz_melhor, modelo_final = melhor_resultado
    plot_holdout(nome_melhor, y_te, score_melhor, matriz_melhor)
    
    # 4) Guardar o modelo final treinado
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    caminho_modelo = MODELS_DIR / "modelo_final.joblib"
    joblib.dump(modelo_final, caminho_modelo)
    
    print(f"\n=== Sucesso! ===")
    print(f"  -> Modelo guardado em: {caminho_modelo.relative_to(ROOT)}")
    print(f"  -> Tabelas em {PASTA_TABELAS.relative_to(ROOT)} e Imagens em {PASTA_FIG.relative_to(ROOT)}")
    
    return resumo, teste

if __name__ == "__main__":
    df_resumo, df_teste = main()
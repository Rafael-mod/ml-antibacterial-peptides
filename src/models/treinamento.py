"""
src/models/treinamento.py

Benchmark de Modelos Baseline (Rigor de Homologia 40% + Incerteza Calibrada):
1. Carrega dados limpos e folds de homologia (meta_train.csv).
2. Executa CV 5-fold por homologia sem vazamento de pré-processamento.
3. Avalia o(s) melhor(es) modelo(s) no teste cego + Conformal Prediction.
4. Salva gráficos, tabelas e o modelo final.
"""

import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

# Garante importação limpa ao executar o script diretamente
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.utils_modelagem import (
    PreprocessadorDinamico,
    avaliar_conformal_prediction,
    calcular_scale_pos_weight,
    carregar_dados_e_cv,
    criar_scoring,
    metricas_holdout,
)

sns.set_theme(style="whitegrid", context="notebook")

PROC_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"
MODELS_DIR = ROOT / "models"
PASTA_FIG = REPORTS_DIR / "figuras_modelagem"
PASTA_TABELAS = REPORTS_DIR / "tabelas"

SEED = 42
N_JOBS = -1
LIMITE_SKEW = 1.0


# ==========================================
# 1. Configuração de Modelos
# ==========================================
def criar_modelos(
    y_tr: pd.Series,
    n_arvores: int = 300,
    usar_balanceamento: bool = True,
) -> Dict[str, Any]:
    """Instancia modelos com pré-processamento interno (zero leakage) e pesos de classe."""
    cw = "balanced" if usar_balanceamento else None
    spw = calcular_scale_pos_weight(y_tr) if usar_balanceamento else 1.0

    prep_base = PreprocessadorDinamico(limite_skew=LIMITE_SKEW, verbose=False)

    svc_base = SVC(kernel="rbf", C=1.0, gamma="scale", class_weight=cw, random_state=SEED)
    svc_calibrado = CalibratedClassifierCV(estimator=svc_base, cv=3, ensemble=False)

    modelos: Dict[str, Any] = {
        "Regressão Logística": Pipeline([
            ("prep", clone(prep_base)),
            ("clf", LogisticRegression(max_iter=3000, class_weight=cw, random_state=SEED)),
        ]),
        "SVM (RBF)": Pipeline([
            ("prep", clone(prep_base)),
            ("clf", svc_calibrado),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=n_arvores, class_weight=cw, n_jobs=N_JOBS, random_state=SEED
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=n_arvores, class_weight=cw, n_jobs=N_JOBS, random_state=SEED
        ),
    }

    if XGBClassifier is not None:
        modelos["XGBoost"] = XGBClassifier(
            n_estimators=n_arvores,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.5,
            scale_pos_weight=spw,
            tree_method="hist",
            eval_metric="logloss",
            n_jobs=N_JOBS,
            random_state=SEED,
        )
    else:
        print("  -> Aviso: XGBoost não instalado. Modelo ignorado.")

    if LGBMClassifier is not None:
        modelos["LightGBM"] = LGBMClassifier(
            n_estimators=n_arvores,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.5,
            class_weight=cw,
            n_jobs=N_JOBS,
            random_state=SEED,
            verbose=-1,
        )
    else:
        print("  -> Aviso: LightGBM não instalado. Modelo ignorado.")

    return modelos


# ==========================================
# 2. Validação Cruzada (Homologia 40%)
# ==========================================
def validacao_cruzada(
    modelos: Dict[str, Any],
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    cv: Any,
    groups: Optional[np.ndarray] = None,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, np.ndarray]]]:
    scoring = criar_scoring()
    resumo, por_dobra = [], {}

    for nome, modelo in modelos.items():
        t0 = time.time()
        res = cross_validate(
            modelo, X_tr, y_tr, cv=cv, groups=groups, scoring=scoring, n_jobs=1
        )
        tempo = round(time.time() - t0, 1)

        linha = {"modelo": nome, "tempo_s": tempo}
        for m in scoring:
            val = res[f"test_{m}"]
            linha[f"{m}_media"], linha[f"{m}_dp"] = val.mean(), val.std()

        resumo.append(linha)
        por_dobra[nome] = {m: res[f"test_{m}"] for m in scoring}

        print(
            f"  -> {nome:<20} MCC: {linha['mcc_media']:.3f} ± {linha['mcc_dp']:.3f} "
            f"| AUC: {linha['roc_auc_media']:.3f} ({tempo}s)"
        )

    df_resumo = pd.DataFrame(resumo).sort_values("mcc_media", ascending=False).reset_index(drop=True)
    return df_resumo, por_dobra


def tabela_formatada(resumo: pd.DataFrame) -> pd.DataFrame:
    t = pd.DataFrame({"modelo": resumo["modelo"]})
    for m in ["mcc", "roc_auc", "pr_auc", "sensibilidade", "especificidade", "acuracia"]:
        t[m] = resumo[f"{m}_media"].map("{:.3f}".format) + " ± " + resumo[f"{m}_dp"].map("{:.3f}".format)
    return t


# ==========================================
# 3. Exportação Visual
# ==========================================
def salvar(nome: str, fig: plt.Figure) -> None:
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
        ax.set_title(f"{titulo} por Dobra (Homologia 40%)")

    fig.tight_layout()
    salvar("01_cv_benchmark.png", fig)


def plot_holdout(nome: str, y_te: pd.Series, score: np.ndarray, matriz: Tuple[int, int, int, int]) -> None:
    tn, fp, fn, tp = matriz
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    fpr, tpr, _ = roc_curve(y_te, score)
    axes[0].plot(fpr, tpr, lw=2, color="#d62728", label=f"AUC = {roc_auc_score(y_te, score):.3f}")
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[0].set(title="Curva ROC (Teste Cego)", xlabel="1 − Especificidade", ylabel="Sensibilidade")
    axes[0].legend(loc="lower right")

    prec, rec, _ = precision_recall_curve(y_te, score)
    axes[1].plot(rec, prec, lw=2, color="#1f77b4", label=f"PR-AUC = {average_precision_score(y_te, score):.3f}")
    axes[1].set(title="Curva Precisão-Recall (Teste Cego)", xlabel="Recall", ylabel="Precisão", ylim=(0, 1.02))
    axes[1].legend(loc="lower left")

    sns.heatmap(
        np.array([[tn, fp], [fn, tp]]),
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Prev. Neg", "Prev. Pos"],
        yticklabels=["Real Neg", "Real Pos"],
        ax=axes[2],
    )
    axes[2].set_title("Matriz de Confusão (Teste Cego)")

    fig.suptitle(f"Desempenho do Modelo Selecionado: {nome}", y=1.05, fontsize=14, fontweight="bold")
    salvar("02_holdout_melhor_modelo.png", fig)


# ==========================================
# 4. Pipeline Principal
# ==========================================
def main(
    n_splits: int = 5,
    n_arvores: int = 300,
    usar_balanceamento: bool = True,
    avaliar_todos_no_teste: bool = False,
    alpha_conformal: float = 0.10,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    print("=== Iniciando Pipeline de Modelação (Homologia 40%) ===")

    X_tr, X_te, y_tr, y_te, cv, groups = carregar_dados_e_cv(PROC_DIR, n_splits=n_splits, seed=SEED)
    print(f"Conjunto de Treino: {X_tr.shape[0]} amostras, {X_tr.shape[1]} features")
    print(f"Conjunto de Teste Cego: {X_te.shape[0]} amostras, {X_te.shape[1]} features")

    # Log único e informativo do pré-processamento (sem vazamento na CV)
    _ = PreprocessadorDinamico(limite_skew=LIMITE_SKEW, verbose=True).fit(X_tr)

    modelos = criar_modelos(y_tr, n_arvores=n_arvores, usar_balanceamento=usar_balanceamento)
    print("\n[1/3] Validação Cruzada por Homologia 40% no Treino:")
    resumo, por_dobra = validacao_cruzada(modelos, X_tr, y_tr, cv=cv, groups=groups)

    PASTA_TABELAS.mkdir(parents=True, exist_ok=True)
    resumo.to_csv(PASTA_TABELAS / "resultados_cv.csv", index=False)

    print("\nResumo da CV (Média ± Desvio Padrão):")
    print(tabela_formatada(resumo).to_string(index=False))
    plot_cv(por_dobra, list(resumo["modelo"]))

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

    # [3/3] Incerteza Calibrada (Conformal Prediction) no Campeão
    print(f"\n[3/3] Conformal Prediction ({int((1 - alpha_conformal) * 100)}% garantia) — {nome_melhor}:")
    res_conf, df_conf = avaliar_conformal_prediction(
        modelo_base=modelos[nome_melhor],
        modelo_treinado=modelo_final,
        X_tr=X_tr,
        y_tr=y_tr,
        X_te=X_te,
        y_te=y_te,
        cv=cv,
        groups=groups,
        alpha=alpha_conformal,
    )
    print(
        f"  -> Cobertura empírica: {res_conf['cobertura_empirica']:.3f} (alvo {res_conf['cobertura_alvo']:.2f}) | "
        f"Alta confiança: {res_conf['pct_alta_confianca']:.1f}% (MCC={res_conf['mcc_alta_confianca']:.3f}) | "
        f"Ambíguos: {res_conf['pct_ambiguo']:.1f}%"
    )
    pd.DataFrame([res_conf]).to_csv(PASTA_TABELAS / "conformal_resumo_baseline.csv", index=False)
    df_conf.to_csv(PASTA_TABELAS / "conformal_predicoes_baseline.csv", index=False)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    caminho_modelo = MODELS_DIR / "modelo_final.joblib"
    joblib.dump(modelo_final, caminho_modelo)

    print("\n=== Sucesso! ===")
    print(f"  -> Modelo salvo em: {caminho_modelo.relative_to(ROOT)}")
    print(f"  -> Tabelas em {PASTA_TABELAS.relative_to(ROOT)} e Imagens em {PASTA_FIG.relative_to(ROOT)}")

    return resumo, teste


if __name__ == "__main__":
    main()
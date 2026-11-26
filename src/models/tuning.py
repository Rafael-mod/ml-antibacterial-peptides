"""
src/models/tuning.py

Busca de hiperparâmetros (RandomizedSearchCV) respeitando homologia 40%,
sem data leakage no pré-processamento e com Conformal Prediction no campeão.
"""

import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import loguniform, randint
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import RandomizedSearchCV, cross_validate
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
PASTA_FIG = REPORTS_DIR / "figuras_tuning"
PASTA_TABELAS = REPORTS_DIR / "tabelas"
MODELS_DIR = ROOT / "models"

SEED = 42
N_JOBS_BUSCA = -1
LIMITE_SKEW = 1.0
MODELOS_ALVO = ("SVM (RBF)", "LightGBM", "XGBoost", "Random Forest", "Extra Trees")


# ==========================================
# 1. Modelos Base e Espaços de Busca
# ==========================================
def criar_estimadores(y_tr: pd.Series, usar_balanceamento: bool = True):
    cw = "balanced" if usar_balanceamento else None
    spw = calcular_scale_pos_weight(y_tr) if usar_balanceamento else 1.0

    est = {
        "SVM (RBF)": Pipeline([
            ("prep", PreprocessadorDinamico(limite_skew=LIMITE_SKEW)),
            ("clf", SVC(kernel="rbf", class_weight=cw, random_state=SEED, cache_size=2000)),
        ]),
        "Random Forest": RandomForestClassifier(class_weight=cw, n_jobs=1, random_state=SEED),
        "Extra Trees": ExtraTreesClassifier(class_weight=cw, n_jobs=1, random_state=SEED),
    }

    if XGBClassifier is not None:
        est["XGBoost"] = XGBClassifier(
            scale_pos_weight=spw,
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=1,
            random_state=SEED,
        )
    if LGBMClassifier is not None:
        est["LightGBM"] = LGBMClassifier(
            class_weight=cw, n_jobs=1, random_state=SEED, verbose=-1
        )

    return est


ESPACOS = {
    "SVM (RBF)": {
        "clf__C": loguniform(0.5, 20.0),
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
        "n_estimators": randint(100, 250),
        "max_depth": [10, 15, None],
        "max_features": ["sqrt", "log2", 0.3],
        "min_samples_leaf": randint(2, 6),
    },
    "Extra Trees": {
        "n_estimators": randint(100, 300),
        "max_depth": [12, 20, None],
        "max_features": ["sqrt", "log2", 0.4],
        "min_samples_leaf": randint(1, 5),
    },
}


def _nativo(v):
    return v.item() if hasattr(v, "item") else v


# ==========================================
# 2. Busca de Hiperparâmetros (Homologia 40%)
# ==========================================
def otimizar(nome, base, X_tr, y_tr, cv, groups, n_iter):
    scoring = {k: v for k, v in criar_scoring().items() if k in ("mcc", "roc_auc", "pr_auc")}
    padrao = cross_validate(base, X_tr, y_tr, cv=cv, groups=groups, scoring=scoring, n_jobs=N_JOBS_BUSCA)

    t0 = time.time()
    busca = RandomizedSearchCV(
        base,
        ESPACOS[nome],
        n_iter=n_iter,
        scoring=scoring,
        refit="mcc",
        cv=cv,
        n_jobs=N_JOBS_BUSCA,
        random_state=SEED,
        error_score="raise",
    )
    busca.fit(X_tr, y_tr, groups=groups)
    i = busca.best_index_
    r = busca.cv_results_

    resumo = {
        "modelo": nome,
        "n_iter": n_iter,
        "tempo_busca_s": round(time.time() - t0, 1),
        "mcc_cv_padrao": padrao["test_mcc"].mean(),
        "mcc_cv_otimizado": r["mean_test_mcc"][i],
        "mcc_cv_otimizado_dp": r["std_test_mcc"][i],
        "roc_auc_cv_padrao": padrao["test_roc_auc"].mean(),
        "roc_auc_cv_otimizado": r["mean_test_roc_auc"][i],
        "pr_auc_cv_padrao": padrao["test_pr_auc"].mean(),
        "pr_auc_cv_otimizado": r["mean_test_pr_auc"][i],
    }
    params = {k: _nativo(v) for k, v in busca.best_params_.items()}

    print(
        f"  {nome:<14} MCC CV (40%): padrão {resumo['mcc_cv_padrao']:.3f} -> otimizado "
        f"{resumo['mcc_cv_otimizado']:.3f} ± {resumo['mcc_cv_otimizado_dp']:.3f}  "
        f"({resumo['tempo_busca_s']}s)"
    )
    return busca, resumo, params


# ==========================================
# 3. Gráficos
# ==========================================
def salvar(nome, fig):
    PASTA_FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(PASTA_FIG / nome, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_padrao_vs_otimizado(resumo):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    for ax, (m, titulo) in zip(axes, [("mcc", "MCC"), ("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC")]):
        d = resumo.melt(
            id_vars="modelo",
            value_vars=[f"{m}_cv_padrao", f"{m}_cv_otimizado"],
            var_name="config",
            value_name="valor",
        )
        d["config"] = d["config"].map({f"{m}_cv_padrao": "Padrão", f"{m}_cv_otimizado": "Otimizado"})
        sns.barplot(data=d, x="modelo", y="valor", hue="config", ax=ax, palette=["#bdbdbd", "#DD8452"])
        ax.set_ylim(max(0, d["valor"].min() - 0.05), min(1, d["valor"].max() + 0.02))
        ax.set(title=f"{titulo} (CV Homologia 40%)", xlabel="")
        ax.tick_params(axis="x", rotation=18)
        ax.legend(title="")
    fig.tight_layout()
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
    for ax in axes:
        ax.legend(loc="lower right" if ax is axes[0] else "lower left")
    fig.suptitle(f"{nome}: padrão vs otimizado", y=1.02, fontsize=14)
    salvar(filename, fig)


# ==========================================
# 4. Execução Principal
# ==========================================
def main(n_splits=5, n_iter=35, usar_balanceamento=True, alpha_conformal=0.10):
    X_tr, X_te, y_tr, y_te, cv, groups = carregar_dados_e_cv(PROC_DIR, n_splits=n_splits, seed=SEED)
    print(f"Treino: {len(X_tr)} amostras | Teste cego: {len(X_te)} amostras")

    base = criar_estimadores(y_tr, usar_balanceamento=usar_balanceamento)
    alvo = [m for m in MODELOS_ALVO if m in base]

    print(f"\nRandomizedSearchCV ({n_iter} comb. x {n_splits} dobras — Homologia 40%):")
    buscas, linhas, melhores = {}, [], {}

    for nome in alvo:
        busca, resumo, params = otimizar(nome, base[nome], X_tr, y_tr, cv, groups, n_iter)
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

        met_pad, s_pad, _ = metricas_holdout(padrao, X_te, y_te)
        met_otm, s_otm, _ = metricas_holdout(otimizado, X_te, y_te)

        linhas_teste.extend([
            {"modelo": nome, "config": "padrão", **met_pad},
            {"modelo": nome, "config": "otimizado", **met_otm},
        ])
        plot_holdout(nome, y_te, s_pad, s_otm, f"02_holdout_{nome.replace(' ', '_')}.png")

    teste = pd.DataFrame(linhas_teste)
    teste.to_csv(PASTA_TABELAS / "resultados_teste_cego_tuning.csv", index=False)
    print("\nDesempenho no Teste Cego:")
    print(teste.round(3).to_string(index=False))

    campeao = resumo.loc[0, "modelo"]
    modelo_final = buscas[campeao].best_estimator_

    # Se o campeão não tiver predict_proba (ex: SVM RBF), calibra respeitando os folds de homologia
    if not hasattr(modelo_final, "predict_proba"):
        print(f"\n  -> Calibrando probabilidades de {campeao} (Platt Scaling) para Conformal Prediction...")
        modelo_base_cal = CalibratedClassifierCV(clone(modelo_final), cv=3, method="sigmoid", ensemble=False)
        modelo_final = clone(modelo_base_cal).fit(X_tr, y_tr)
    else:
        modelo_base_cal = clone(modelo_final)

    res_conf, df_conf = avaliar_conformal_prediction(
        modelo_base=modelo_base_cal,
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
        f"\n[Conformal Prediction — {campeao} | alvo {res_conf['cobertura_alvo']:.0%}]: "
        f"Cobertura={res_conf['cobertura_empirica']:.3f} | "
        f"Alta confiança={res_conf['pct_alta_confianca']:.1f}% (MCC={res_conf['mcc_alta_confianca']:.3f}) | "
        f"Ambíguos={res_conf['pct_ambiguo']:.1f}%"
    )
    pd.DataFrame([res_conf]).to_csv(PASTA_TABELAS / "conformal_resumo_otimizado.csv", index=False)
    df_conf.to_csv(PASTA_TABELAS / "conformal_predicoes_otimizado.csv", index=False)

    caminho = MODELS_DIR / "modelo_final_otimizado.joblib"
    joblib.dump(modelo_final, caminho)
    print(f"\n[!] Campeão ({campeao}) salvo em: {caminho.relative_to(ROOT)}")

    return resumo, teste


if __name__ == "__main__":
    main()
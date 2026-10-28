"""
src/models/shap_analysis.py

Análise de explicabilidade do modelo campeão (SVM RBF) utilizando SHAP.
Gera os gráficos Beeswarm, Bar Plot e a tabela com as Top 30 features.
"""

import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from pathlib import Path

# ==========================================
# Constantes e Estrutura de Diretórios
# ==========================================
ROOT = Path(__file__).resolve().parents[2]
PROC_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figuras_modelagem"
TAB_DIR = REPORTS_DIR / "tabelas"

def executar_analise_shap():
    print("=== Iniciando Análise SHAP para o Modelo Campeão (SVM RBF) ===")

    # 1. Carregar os dados limpos (já filtrados pelo pipeline Anti-Leakage)
    caminho_treino = PROC_DIR / "X_train_limpo.csv"
    caminho_teste = PROC_DIR / "X_test_limpo.csv"
    
    if not caminho_treino.exists() or not caminho_teste.exists():
        raise FileNotFoundError("Matrizes 'X_train_limpo.csv' ou 'X_test_limpo.csv' não encontradas.")

    X_train = pd.read_csv(caminho_treino)
    X_test = pd.read_csv(caminho_teste)
    print(f"  -> Dados carregados. Features a analisar: {X_test.shape[1]}")

    # 2. Carregar o modelo otimizado
    caminho_modelo = MODELS_DIR / "modelo_final_otimizado.joblib"
    if not caminho_modelo.exists():
        # Fallback para nome genérico caso não encontre o otimizado
        caminho_modelo = MODELS_DIR / "modelo_final.joblib"
        
    pipeline = joblib.load(caminho_modelo)
    print(f"  -> Modelo {type(pipeline).__name__} carregado com sucesso.")

    # 3. Tratar Pipeline vs Classificador Direto
    # Se o modelo for um Pipeline do Scikit-Learn (ex: StandardScaler + SVC):
    if hasattr(pipeline, 'named_steps'):
        prep = pipeline.named_steps.get('prep', None)
        clf = pipeline.named_steps.get('clf', pipeline.steps[-1][1])
        if prep is not None:
            X_train_trans = pd.DataFrame(prep.transform(X_train), columns=X_train.columns)
            X_test_trans = pd.DataFrame(prep.transform(X_test), columns=X_test.columns)
        else:
            X_train_trans, X_test_trans = X_train, X_test
    else:
        clf = pipeline
        X_train_trans, X_test_trans = X_train, X_test

    # Função de predição (probabilidade da classe 1 - Ativo)
    if hasattr(clf, "predict_proba"):
        predict_fn = lambda x: clf.predict_proba(x)[:, 1]
    else:
        # Fallback se o SVM não foi treinado com probability=True
        print("  -> Aviso: predict_proba não encontrado. Usando decision_function.")
        predict_fn = lambda x: clf.decision_function(x)

    # 4. Resumir Background Data para Acelerar o KernelExplainer
    # O k-means reduz o dataset de treino para K centroides representativos
    print("  -> Resumindo background data com shap.kmeans (k=50) para otimizar tempo...")
    background_data = shap.kmeans(X_train_trans, 50)

    # 5. Criar Explainer e Calcular SHAP Values
    print(f"  -> Calculando valores SHAP no teste cego ({X_test.shape[0]} amostras)...")
    explainer = shap.KernelExplainer(predict_fn, background_data)
    
    # Suprime os warnings de progresso do KernelExplainer
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shap_values_obj = explainer.shap_values(X_test_trans)

    # Extrai array da classe positiva dependendo do formato de retorno da versão do SHAP
    shap_vals = shap_values_obj[1] if isinstance(shap_values_obj, list) else shap_values_obj

    # 6. Criar diretórios de destino
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)

    # 7. Salvar Tabela de Importância Global (Top 30)
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    df_shap = pd.DataFrame({
        'feature': X_test.columns,
        'mean_abs_shap': mean_abs_shap
    }).sort_values(by='mean_abs_shap', ascending=False).reset_index(drop=True)

    df_shap['rank'] = df_shap.index + 1
    
    caminho_csv = TAB_DIR / "shap_top30_features.csv"
    df_shap.head(30).to_csv(caminho_csv, index=False)
    print(f"  -> Tabela de relevância salva em: {caminho_csv.relative_to(ROOT)}")

    # 8. Gerar e Salvar Gráfico Beeswarm (Sumário de Impacto e Direção)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_vals, X_test_trans, feature_names=X_test.columns, max_display=20, show=False)
    plt.title("SHAP Summary Plot (Top 20 Features) - SVM RBF", fontsize=14, pad=15)
    plt.tight_layout()
    caminho_fig_beeswarm = FIG_DIR / "shap_beeswarm_svm.png"
    plt.savefig(caminho_fig_beeswarm, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> Beeswarm plot salvo em: {caminho_fig_beeswarm.relative_to(ROOT)}")

    # 9. Gerar e Salvar Gráfico de Barras (Importância Absoluta Média)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_vals, X_test_trans, feature_names=X_test.columns, plot_type='bar', max_display=20, show=False)
    plt.title("Feature Importance SHAP (Mean |SHAP Value|) - SVM RBF", fontsize=14, pad=15)
    plt.tight_layout()
    caminho_fig_bar = FIG_DIR / "shap_bar_svm.png"
    plt.savefig(caminho_fig_bar, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> Bar plot salvo em: {caminho_fig_bar.relative_to(ROOT)}")

    print("\n=== Análise SHAP Finalizada com Sucesso! ===")

if __name__ == "__main__":
    executar_analise_shap()
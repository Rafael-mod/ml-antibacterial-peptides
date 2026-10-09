"""
src/models/shap_analysis.py

Análise de explicabilidade do modelo campeão utilizando SHAP.
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
PASTA_FIG = REPORTS_DIR / "figuras_shap"

def principal():
    # 1. Carregar os dados de treino para servir de base explicativa
    caminho_dados = PROC_DIR / "dataset_treino_limpo.csv"
    if not caminho_dados.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho_dados}")

    df = pd.read_csv(caminho_dados)
    
    # Isolar apenas as features (exatamente como foi feito no treino)
    X = df.drop(columns=["sequence", "label", "classe"], errors="ignore")
    X = X.select_dtypes(include=[np.number])

    # 2. Carregar o modelo otimizado
    caminho_modelo = MODELS_DIR / "modelo_final.joblib"
    modelo = joblib.load(caminho_modelo)
    print(f"Modelo {type(modelo).__name__} carregado com sucesso.")

    # 3. Inicializar o Explainer do SHAP
    # Como o LightGBM é um modelo baseado em árvores, usamos o TreeExplainer (extremamente rápido e exato)
    explainer = shap.TreeExplainer(modelo)
    
    # Calcular os valores SHAP para todas as amostras
    # Retorna um objeto de Explicação contendo valores, dados base e nomes de features
    print("Calculando valores SHAP (isso pode levar alguns segundos)...")
    shap_values = explainer(X)

    # Nota: Em classificadores binários, algumas versões do SHAP retornam valores para ambas as classes.
    # Se shape for 3D (amostras, features, classes), isolamos a classe 1 (positiva)
    if len(shap_values.shape) == 3:
        shap_values = shap_values[:, :, 1]

    # 4. Gerar e salvar o gráfico Summary Plot (Beeswarm)
    PASTA_FIG.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(12, 8))
    # O beeswarm plot mostra a densidade das amostras e a direção do impacto (cores representam o valor real da feature)
    shap.plots.beeswarm(shap_values, max_display=20, show=False)
    
    plt.title("Impacto das Top 20 Features na Saída do Modelo (SHAP Values)", fontsize=14)
    plt.tight_layout()
    
    caminho_figura = PASTA_FIG / "shap_summary_beeswarm.png"
    plt.savefig(caminho_figura, dpi=300, bbox_inches="tight")
    print(f"Gráfico SHAP salvo em: {caminho_figura.relative_to(ROOT)}")
    
    plt.show()

    # 5. Bônus: Extrair a importância global média e salvar em CSV
    importancia_global = np.abs(shap_values.values).mean(axis=0)
    df_importancia = pd.DataFrame({
        "Feature": X.columns,
        "Importancia_Media_Absoluta": importancia_global
    }).sort_values(by="Importancia_Media_Absoluta", ascending=False).reset_index(drop=True)
    
    caminho_csv = PASTA_FIG / "shap_feature_importance.csv"
    df_importancia.to_csv(caminho_csv, index=False)
    print(f"Tabela de importância de features salva em: {caminho_csv.relative_to(ROOT)}")


if __name__ == "__main__":
    principal()
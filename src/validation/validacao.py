"""
src/models/validacao.py

Validação de Estabilidade do Modelo Campeão (10x Random Seeds)
1. Carrega o dataset limpo e o modelo campeão.
2. Executa 10 iterações de Train/Test Split com sementes diferentes.
3. Plota a Curva ROC Média (com desvio padrão/sombra de confiança).
4. Gera Boxplots comparando métricas de Treino vs Teste para provar ausência de overfitting.
5. Salva os resultados detalhados em CSV.

Uso:
    python src/models/validacao.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.base import clone
from sklearn.metrics import roc_curve, auc, accuracy_score, matthews_corrcoef, f1_score

# ==========================================
# Configurações de Diretórios
# ==========================================
ROOT = Path(__file__).resolve().parents[2]
PROC_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figuras_modelagem"
TAB_DIR = REPORTS_DIR / "tabelas"

# ==========================================
# Execução da Validação
# ==========================================
def executar_validacao():
    print("=== Validação de Estabilidade (10x Random Seeds) ===")

    # 1. Carregar Dados
    X_csv = PROC_DIR / "X_train_limpo.csv"
    y_csv = PROC_DIR / "y_train.csv"
    
    if not X_csv.exists() or not y_csv.exists():
        raise FileNotFoundError("Bases limpas não encontradas. Execute o tratamento de colunas primeiro.")

    X = pd.read_csv(X_csv)
    y = pd.read_csv(y_csv).squeeze()
    
    # 2. Carregar Modelo Campeão
    caminho_modelo = MODELS_DIR / "modelo_final.joblib"
    if not caminho_modelo.exists():
        raise FileNotFoundError(f"Modelo não encontrado em: {caminho_modelo}")
        
    modelo_base = joblib.load(caminho_modelo)
    print(f"  -> Modelo {type(modelo_base).__name__} carregado. Iniciando 10 rodadas...\n")

    # 3. Preparar Estruturas de Armazenamento
    seeds = [42, 123, 999, 7, 88, 2023, 10, 55, 333, 777]
    resultados = []
    
    # Para a Curva ROC Média
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    fig_roc, ax_roc = plt.subplots(figsize=(8, 6))

    # 4. Loop de Validação
    for i, seed in enumerate(seeds):
        # Separar dados variando a seed
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
        
        # Clonar o modelo para garantir que ele comece do zero em cada iteração
        modelo = clone(modelo_base)
        modelo.fit(X_tr, y_tr)
        
        # Previsões Treino
        y_tr_pred = modelo.predict(X_tr)
        
        # Previsões Teste
        y_te_pred = modelo.predict(X_te)
        
        # Probabilidades para ROC (fallback para decision_function se predict_proba não existir)
        if hasattr(modelo, "predict_proba"):
            y_te_prob = modelo.predict_proba(X_te)[:, 1]
        else:
            y_te_prob = modelo.decision_function(X_te)
            
        # Calcular Métricas Teste
        acc_te = accuracy_score(y_te, y_te_pred)
        mcc_te = matthews_corrcoef(y_te, y_te_pred)
        f1_te = f1_score(y_te, y_te_pred)
        
        # Calcular Métricas Treino
        acc_tr = accuracy_score(y_tr, y_tr_pred)
        mcc_tr = matthews_corrcoef(y_tr, y_tr_pred)
        f1_tr = f1_score(y_tr, y_tr_pred)
        
        # Curva ROC da iteração
        fpr, tpr, _ = roc_curve(y_te, y_te_prob)
        roc_auc = auc(fpr, tpr)
        
        # Interpolação para a curva média
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(roc_auc)
        
        # Plotar curva individual mais fraca ao fundo
        ax_roc.plot(fpr, tpr, color='gray', alpha=0.15)
        
        resultados.append({
            'Seed': seed,
            'Train_ACC': acc_tr, 'Test_ACC': acc_te,
            'Train_MCC': mcc_tr, 'Test_MCC': mcc_te,
            'Train_F1': f1_tr, 'Test_F1': f1_te,
            'Test_AUC': roc_auc
        })
        
        print(f"  -> Iteração {i+1}/10 (Seed {seed}): Test AUC = {roc_auc:.3f} | Test MCC = {mcc_te:.3f}")

    # 5. Processar Resultados e Salvar Tabela
    df_res = pd.DataFrame(resultados)
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    df_res.to_csv(TAB_DIR / "validacao_10x_resultados.csv", index=False)
    
    print("\nMÉDIA FINAL (± Desvio Padrão):")
    print(f"  -> AUC: {df_res['Test_AUC'].mean():.3f} ± {df_res['Test_AUC'].std():.3f}")
    print(f"  -> MCC: {df_res['Test_MCC'].mean():.3f} ± {df_res['Test_MCC'].std():.3f}")
    
    # 6. Finalizar e Salvar Plot ROC Média
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    
    ax_roc.plot(mean_fpr, mean_tpr, color='b',
             label=r'Curva Média (AUC = %0.3f $\pm$ %0.3f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)
    
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax_roc.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 Desvio Padrão')
    
    ax_roc.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=.8)
    ax_roc.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
               title="Curva ROC Média (10x Random Seeds)",
               xlabel="Taxa de Falsos Positivos",
               ylabel="Taxa de Verdadeiros Positivos")
    ax_roc.legend(loc="lower right")
    
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig_roc.savefig(FIG_DIR / "10_roc_media_10x.png", dpi=300, bbox_inches="tight")
    plt.close(fig_roc)

    # 7. Plot de Estabilidade (Boxplot Treino vs Teste)
    # Ajustando o melt correto para as colunas que temos:
    df_melt = df_res.melt(id_vars=['Seed'], 
                          value_vars=['Train_ACC', 'Test_ACC', 'Train_MCC', 'Test_MCC', 'Train_F1', 'Test_F1'],
                          var_name='Metrica_Base', value_name='Score')
                          
    df_melt['Conjunto'] = df_melt['Metrica_Base'].apply(lambda x: 'Treino' if 'Train' in x else 'Teste')
    df_melt['Métrica'] = df_melt['Metrica_Base'].apply(lambda x: x.split('_')[1])

    sns.set_theme(style="whitegrid", context="paper")
    fig_box, ax_box = plt.subplots(figsize=(10, 6))
    
    sns.boxplot(data=df_melt, x='Métrica', y='Score', hue='Conjunto', palette=["#4C72B0", "#DD8452"], ax=ax_box)
    sns.stripplot(data=df_melt, x='Métrica', y='Score', hue='Conjunto', dodge=True, color='black', alpha=0.4, ax=ax_box, legend=False)
    
    ax_box.set_title("Estabilidade do Modelo: Treino vs Teste (10 Sementes)", fontsize=14, pad=15)
    ax_box.set_ylim(0, 1.05)
    ax_box.set_xlabel("")
    ax_box.set_ylabel("Pontuação")
    ax_box.legend(title="", loc="lower right")
    
    fig_box.savefig(FIG_DIR / "11_estabilidade_treino_teste_10x.png", dpi=300, bbox_inches="tight")
    plt.close(fig_box)

    print(f"\n=== Concluído! Gráficos salvos em: {FIG_DIR.relative_to(ROOT)} ===")

if __name__ == "__main__":
    executar_validacao()
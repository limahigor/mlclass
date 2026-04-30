import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from .config import FIG_DIR

sns.set_theme(style='whitegrid', context='talk')


def generate_plots(df, numeric_cols, corr, cv_df_round, y_test, y_pred, y_values, best_model_name, summary):
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    order = sorted(y_values.unique())
    sns.countplot(data=df, x='type', order=order, hue='type', palette='viridis', legend=False)
    plt.title('Distribuição das classes no conjunto de treino')
    plt.xlabel('Classe (type)')
    plt.ylabel('Quantidade')
    plt.tight_layout()
    plt.savefig(FIG_DIR / '01_distribuicao_classes.png', dpi=180)
    plt.close()

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    for ax, col in zip(axes.flat, numeric_cols):
        sns.histplot(data=df, x=col, hue='type', kde=True, stat='density', common_norm=False, ax=ax)
        ax.set_title(f'Distribuição de {col}')
    plt.tight_layout()
    plt.savefig(FIG_DIR / '02_histogramas_por_classe.png', dpi=180)
    plt.close()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', square=True)
    plt.title('Correlação entre atributos numéricos e classe codificada')
    plt.tight_layout()
    plt.savefig(FIG_DIR / '03_matriz_correlacao.png', dpi=180)
    plt.close()

    pairplot_df = df[['length', 'diameter', 'whole_weight', 'shell_weight', 'type']].copy()
    pairplot = sns.pairplot(pairplot_df, hue='type', corner=True, diag_kind='hist', plot_kws={'alpha': 0.6, 's': 30})
    pairplot.fig.suptitle('Dispersão entre atributos representativos', y=1.02)
    pairplot.savefig(FIG_DIR / '04_pairplot_atributos.png', dpi=160)
    plt.close('all')

    plt.figure(figsize=(9, 7))
    sns.scatterplot(data=df, x='length', y='diameter', hue='type', alpha=0.7)
    plt.title('Dispersão: length vs diameter')
    plt.tight_layout()
    plt.savefig(FIG_DIR / '05_dispersao_length_diameter.png', dpi=180)
    plt.close()

    plt.figure(figsize=(9, 7))
    sns.scatterplot(data=df, x='whole_weight', y='shell_weight', hue='type', alpha=0.7)
    plt.title('Dispersão: whole_weight vs shell_weight')
    plt.tight_layout()
    plt.savefig(FIG_DIR / '06_dispersao_whole_shell_weight.png', dpi=180)
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=df.melt(id_vars='type', value_vars=numeric_cols, var_name='feature', value_name='value'),
        x='feature', y='value', hue='type'
    )
    plt.xticks(rotation=45, ha='right')
    plt.title('Boxplots por atributo e classe')
    plt.tight_layout()
    plt.savefig(FIG_DIR / '07_boxplots_por_classe.png', dpi=180)
    plt.close()

    plt.figure(figsize=(10, 5))
    sns.barplot(data=cv_df_round, x='accuracy_mean', y='model', hue='model', palette='crest', legend=False)
    plt.xlim(0, 1)
    plt.xlabel('Acurácia média (CV 5-fold)')
    plt.ylabel('Modelo')
    plt.title('Comparação de modelos por validação cruzada')
    plt.tight_layout()
    plt.savefig(FIG_DIR / '08_comparacao_modelos_cv.png', dpi=180)
    plt.close()

    cm = confusion_matrix(y_test, y_pred, labels=sorted(y_values.unique()))
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=sorted(y_values.unique()), yticklabels=sorted(y_values.unique()))
    plt.xlabel('Predito')
    plt.ylabel('Real')
    plt.title(f'Matriz de confusão do holdout - {best_model_name}')
    plt.tight_layout()
    plt.savefig(FIG_DIR / '09_matriz_confusao_holdout.png', dpi=180)
    plt.close()

    if best_model_name == 'random_forest' and 'random_forest_feature_importances' in summary:
        top_imp = pd.Series(summary['random_forest_feature_importances']).sort_values(ascending=True)
        plt.figure(figsize=(10, 6))
        top_imp.plot(kind='barh', color='teal')
        plt.title('Importância das variáveis no Random Forest')
        plt.xlabel('Importância')
        plt.tight_layout()
        plt.savefig(FIG_DIR / '10_importancia_variaveis_rf.png', dpi=180)
        plt.close()

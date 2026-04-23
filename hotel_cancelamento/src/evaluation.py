
"""
evaluation.py — Feature Importance e análise comparativa final
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import COLORS, OUTPUT_DIR


def _save(fig, fname):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig.savefig(os.path.join(OUTPUT_DIR, fname), bbox_inches="tight", dpi=120)
    plt.show()
    plt.close(fig)


# Feature Importance (RF + GB)

def plot_feature_importance(fitted_models: dict, feature_names: list) -> None:
    """Top-20 features para Random Forest e Gradient Boosting lado a lado."""
    tree_models = {k: v for k, v in fitted_models.items() if k != "Logistic Regression"}
    if len(tree_models) < 2:
        print("Modelos de árvore insuficientes para o gráfico de feature importance.")
        return

    nomes = list(tree_models.keys())
    cores = [COLORS["green"], COLORS["orange"]]

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle("Feature Importance — Top 20", fontsize=14, fontweight="bold")

    for ax, nome, cor in zip(axes, nomes, cores):
        modelo = tree_models[nome]
        importances = pd.Series(modelo.feature_importances_, index=feature_names)
        top20 = importances.sort_values(ascending=False).head(20)
        top20_sorted = top20.sort_values(ascending=True)

        bar_colors = [cor if v >= top20.mean() else "#bdc3c7" for v in top20_sorted.values]
        top20_sorted.plot(kind="barh", ax=ax, color=bar_colors, edgecolor="white", linewidth=0.8)
        ax.set_title(f"{nome} — Top 20")
        ax.set_xlabel("Importância")
        ax.axvline(top20.mean(), color="red", linestyle="--", alpha=0.6,
                   label=f"Média ({top20.mean():.4f})")
        ax.legend(fontsize=9)

    plt.tight_layout()
    _save(fig, "13_feature_importance.png")


# Tabela comparativa 

def build_results_table(all_metrics: list, cv_results: dict) -> pd.DataFrame:
    """
    Monta DataFrame com as métricas de todos os modelos.

    Returns
    -------
    pd.DataFrame com colunas: Modelo, Acurácia, Precisão, Recall, F1-Score, AUC-ROC, AUC CV
    """
    rows = []
    for m in all_metrics:
        rows.append({
            "Modelo":       m["modelo"],
            "Acurácia":     m["acuracia"],
            "Precisão":     m["precisao"],
            "Recall":       m["recall"],
            "F1-Score":     m["f1"],
            "AUC-ROC":      m["auc"],
            "AUC CV (média)": cv_results[m["modelo"]]["auc_mean"],
        })
    return pd.DataFrame(rows)


def print_results_table(df_resultados: pd.DataFrame) -> None:
    """Imprime tabela formatada e identifica o melhor modelo."""
    df_fmt = df_resultados.copy()
    for col in ["Acurácia", "Precisão", "Recall", "F1-Score", "AUC-ROC", "AUC CV (média)"]:
        df_fmt[col] = df_fmt[col].apply(lambda x: f"{x:.4f}")

    print("\n" + "=" * 85)
    print("TABELA COMPARATIVA — DESAFIO 07 | CANCELAMENTO DE HOTEL")
    print("=" * 85)
    print(df_fmt.to_string(index=False))
    print("=" * 85)

    melhor_f1  = df_resultados.loc[df_resultados["F1-Score"].idxmax()]
    melhor_auc = df_resultados.loc[df_resultados["AUC-ROC"].idxmax()]
    print(f"\n Melhor F1-Score:  {melhor_f1['Modelo']} ({melhor_f1['F1-Score']:.4f})")
    print(f" Melhor AUC-ROC:   {melhor_auc['Modelo']} ({melhor_auc['AUC-ROC']:.4f})")


# Gráfico comparativo final 

def plot_comparacao_final(df_resultados: pd.DataFrame) -> None:
    """Barras comparativas + radar chart."""
    metricas  = ["Acurácia", "Precisão", "Recall", "F1-Score", "AUC-ROC"]
    x         = np.arange(len(metricas))
    width     = 0.25
    cores     = [COLORS["blue"], COLORS["green"], COLORS["orange"]]
    modelos   = df_resultados["Modelo"].tolist()

    fig = plt.figure(figsize=(16, 6))
    fig.suptitle("Comparação Final dos Modelos — Desafio 07", fontsize=14, fontweight="bold")

    ax1 = fig.add_subplot(1, 2, 1)
    for i, (nome, cor) in enumerate(zip(modelos, cores)):
        vals = df_resultados[df_resultados["Modelo"] == nome][metricas].values.flatten()
        bars = ax1.bar(x + i * width, vals, width, label=nome, color=cor, edgecolor="white")
        for bar, val in zip(bars, vals):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                     f"{val:.3f}", ha="center", fontsize=7.5, fontweight="bold", rotation=70)
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(metricas)
    ax1.set_ylim(0, 1.18)
    ax1.set_ylabel("Valor")
    ax1.set_title("Comparação por Métrica")
    ax1.legend(fontsize=9)
    ax1.axhline(0.8, color="red", linestyle="--", alpha=0.4, linewidth=1)

    # Radar
    angles = np.linspace(0, 2 * np.pi, len(metricas), endpoint=False).tolist()
    angles += angles[:1]
    ax2 = fig.add_subplot(1, 2, 2, polar=True)
    ax2.set_facecolor("#f8fcff")

    for nome, cor in zip(modelos, cores):
        vals = df_resultados[df_resultados["Modelo"] == nome][metricas].values.flatten().tolist()
        vals += vals[:1]
        ax2.plot(angles, vals, "o-", linewidth=2, color=cor, label=nome)
        ax2.fill(angles, vals, alpha=0.1, color=cor)

    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(metricas, fontsize=10)
    ax2.set_ylim(0, 1)
    ax2.set_title("Radar Chart", fontsize=12, pad=15)
    ax2.legend(loc="upper right", bbox_to_anchor=(1.4, 1.1), fontsize=9)
    ax2.grid(True, alpha=0.4)

    plt.tight_layout()
    _save(fig, "14_comparacao_final.png")


# Feature importance do melhor modelo

def plot_best_model_importance(best_model, best_name: str, feature_names: list) -> None:
    """Importância e acumulada das features do melhor modelo."""
    importances = pd.Series(best_model.feature_importances_, index=feature_names)
    top15 = importances.sort_values(ascending=False).head(15)
    top15_sorted = top15.sort_values(ascending=True)

    all_imp = importances.sort_values(ascending=False)
    cumulative = np.cumsum(all_imp.values / all_imp.sum())
    n_80 = int(np.argmax(cumulative >= 0.80)) + 1

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"Análise Detalhada — {best_name}", fontsize=14, fontweight="bold")

    palette = sns.color_palette("viridis", 15)
    top15_sorted.plot(kind="barh", ax=axes[0], color=palette[::-1], edgecolor="white")
    axes[0].set_title("Top 15 Features Mais Importantes")
    axes[0].set_xlabel("Importância (Gini)")
    for i, val in enumerate(top15_sorted.values):
        axes[0].text(val + 0.001, i, f"{val:.4f}", va="center", fontsize=9)

    axes[1].plot(range(1, len(cumulative) + 1), cumulative, "o-",
                 color=COLORS["purple"], linewidth=2, markersize=4)
    axes[1].axhline(0.80, color="red", linestyle="--", alpha=0.7, label="80% da importância")
    axes[1].axvline(n_80, color=COLORS["orange"], linestyle="--", alpha=0.7,
                    label=f"{n_80} features")
    axes[1].fill_between(range(1, len(cumulative) + 1), cumulative, alpha=0.15, color=COLORS["purple"])
    axes[1].set_title("Importância Acumulada")
    axes[1].set_xlabel("Número de Features")
    axes[1].set_ylabel("Importância Acumulada")
    axes[1].legend()
    axes[1].grid(True, alpha=0.4)

    plt.tight_layout()
    _save(fig, "15_feature_importance_final.png")

    print(f"\n{n_80} features respondem por 80% da importância.")
    print("\nTop 5 features mais importantes:")
    for feat, imp in top15.head(5).items():
        print(f"  {feat}: {imp:.4f} ({imp * 100:.2f}%)")

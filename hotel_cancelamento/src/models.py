
"""
models.py — Treinamento e avaliação dos classificadores
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay, classification_report,
)
from typing import Dict, Any
import os

from src.config import RANDOM_STATE, COLORS, OUTPUT_DIR


#  Fábrica de modelos 

def build_models() -> Dict[str, Any]:
    """Retorna dicionário {nome: instância} com os três classificadores."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE, solver="lbfgs"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=8, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=4, random_state=RANDOM_STATE
        ),
    }


# Avaliação de um único modelo 

def evaluate_model(model, X_train, y_train, X_test, y_test, nome: str) -> Dict[str, Any]:
    """
    Treina `model` e avalia no conjunto de teste.

    Returns
    -------
    dict com métricas, predições e probabilidades.
    """
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "modelo":    nome,
        "acuracia":  accuracy_score(y_test, y_pred),
        "precisao":  precision_score(y_test, y_pred),
        "recall":    recall_score(y_test, y_pred),
        "f1":        f1_score(y_test, y_pred),
        "auc":       roc_auc_score(y_test, y_prob),
        "y_pred":    y_pred,
        "y_prob":    y_prob,
    }

    print(f"\n=== {nome.upper()} ===")
    for k, v in metrics.items():
        if k not in ("modelo", "y_pred", "y_prob"):
            print(f"  {k.capitalize():<10}: {v:.4f}")
    print()
    print(classification_report(y_test, y_pred, target_names=["Mantida", "Cancelada"]))
    return metrics


def _save(fig, fname):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig.savefig(os.path.join(OUTPUT_DIR, fname), bbox_inches="tight", dpi=120)
    plt.show()
    plt.close(fig)


#  Visualizações por modelo 

def plot_model_evaluation(metrics: Dict, y_test, cmap: str, fname: str) -> None:
    """Matriz de confusão + curva ROC para um modelo."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{metrics['modelo']} — Avaliação", fontsize=14, fontweight="bold")

    ConfusionMatrixDisplay.from_predictions(
        y_test, metrics["y_pred"],
        display_labels=["Mantida", "Cancelada"],
        cmap=cmap, ax=axes[0], colorbar=False,
    )
    axes[0].set_title("Matriz de Confusão")

    fpr, tpr, _ = roc_curve(y_test, metrics["y_prob"])
    color = list(COLORS.values())[list(build_models().keys()).index(metrics["modelo"]) % len(COLORS)]
    axes[1].plot(fpr, tpr, color=color, linewidth=2.5, label=f"AUC = {metrics['auc']:.3f}")
    axes[1].plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.6, label="Aleatório")
    axes[1].fill_between(fpr, tpr, alpha=0.1, color=color)
    axes[1].set_title("Curva ROC")
    axes[1].set_xlabel("Taxa de Falsos Positivos")
    axes[1].set_ylabel("Taxa de Verdadeiros Positivos")
    axes[1].legend()
    axes[1].set_xlim([0, 1])
    axes[1].set_ylim([0, 1.02])

    plt.tight_layout()
    _save(fig, fname)


#  Validação cruzada 

def cross_validate_models(fitted_models: Dict, X_cv, y_cv) -> Dict[str, Dict]:
    """
    Stratified K-Fold (k=5) para todos os modelos.

    Returns
    -------
    cv_results : {nome: {acc_mean, f1_mean, auc_mean, ...}}
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_results = {}
    print("\n=== VALIDAÇÃO CRUZADA ESTRATIFICADA (k=5) ===")

    for nome, modelo in fitted_models.items():
        scores = {
            "acc":  cross_val_score(modelo, X_cv, y_cv, cv=skf, scoring="accuracy"),
            "f1":   cross_val_score(modelo, X_cv, y_cv, cv=skf, scoring="f1"),
            "auc":  cross_val_score(modelo, X_cv, y_cv, cv=skf, scoring="roc_auc"),
        }
        cv_results[nome] = {
            "acc_mean": scores["acc"].mean(), "acc_std": scores["acc"].std(),
            "f1_mean":  scores["f1"].mean(),  "f1_std":  scores["f1"].std(),
            "auc_mean": scores["auc"].mean(), "auc_std": scores["auc"].std(),
        }
        r = cv_results[nome]
        print(f"\n{nome}:")
        print(f"  Acurácia: {r['acc_mean']:.4f} ± {r['acc_std']:.4f}")
        print(f"  F1-Score: {r['f1_mean']:.4f} ± {r['f1_std']:.4f}")
        print(f"  AUC-ROC:  {r['auc_mean']:.4f} ± {r['auc_std']:.4f}")

    return cv_results


def plot_kfold(cv_results: Dict) -> None:
    """Barras com desvio padrão para acc, f1 e auc."""
    nomes = list(cv_results.keys())
    cores = [COLORS["blue"], COLORS["green"], COLORS["orange"]]
    metrics_info = [
        ("acc_mean", "acc_std", "Acurácia"),
        ("f1_mean",  "f1_std",  "F1-Score"),
        ("auc_mean", "auc_std", "AUC-ROC"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Comparação via Stratified K-Fold (k=5)", fontsize=14, fontweight="bold")

    for ax, (mean_k, std_k, title) in zip(axes, metrics_info):
        means = [cv_results[n][mean_k] for n in nomes]
        stds  = [cv_results[n][std_k]  for n in nomes]
        bars = ax.bar(nomes, means, color=cores, edgecolor="white", linewidth=1.5)
        ax.errorbar(nomes, means, yerr=stds, fmt="none", color="black", capsize=6, linewidth=2)
        ax.set_title(title)
        ax.set_ylabel(title)
        ax.set_ylim(min(means) * 0.9, 1.05)
        ax.tick_params(axis="x", rotation=15)
        for bar, val, std in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 0.005,
                    f"{val:.3f}", ha="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    _save(fig, "11_kfold_comparacao.png")


# Curvas ROC comparativas 

def plot_roc_comparativo(all_metrics: list, y_test) -> None:
    """Curvas ROC sobrepostas para todos os modelos."""
    cores_list = [COLORS["blue"], COLORS["green"], COLORS["orange"]]
    fig, ax = plt.subplots(figsize=(9, 7))

    for metrics, cor in zip(all_metrics, cores_list):
        fpr, tpr, _ = roc_curve(y_test, metrics["y_prob"])
        ax.plot(fpr, tpr, linewidth=2.5, color=cor,
                label=f"{metrics['modelo']} (AUC = {metrics['auc']:.3f})")
        ax.fill_between(fpr, tpr, alpha=0.07, color=cor)

    ax.plot([0, 1], [0, 1], "k--", linewidth=1.2, alpha=0.6, label="Aleatório (AUC = 0.500)")
    ax.set_title("Curvas ROC — Comparação de Modelos", fontsize=14, fontweight="bold")
    ax.set_xlabel("Taxa de Falsos Positivos")
    ax.set_ylabel("Taxa de Verdadeiros Positivos")
    ax.legend(loc="lower right", fontsize=11)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.grid(True, alpha=0.35)

    plt.tight_layout()
    _save(fig, "12_roc_comparativo.png")

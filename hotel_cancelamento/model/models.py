"""
models.py — Treinamento e avaliação dos classificadores
"""

import os
from typing import Any, Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.config import COLORS, OUTPUT_DIR, RANDOM_STATE


# Mapeamento estável nome → cor (evita recriar modelos só para pegar a cor)
_MODEL_COLORS: Dict[str, str] = {
    "Logistic Regression": COLORS["blue"],
    "Random Forest":       COLORS["green"],
    "Gradient Boosting":   COLORS["orange"],
}
def build_models() -> Dict[str, Any]:
    """Retorna dicionário {nome: instância} com os três classificadores.

    Os modelos são sempre **não treinados** para que possam ser usados tanto
    em treino direto quanto em validação cruzada (que re-fita internamente).
    """
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


def evaluate_model(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    nome: str,
) -> Dict[str, Any]:
    """Treina *model* e avalia no conjunto de teste.

    Parameters
    ----------
    model:
        Estimador scikit-learn (não treinado).
    X_train, y_train:
        Dados de treino.
    X_test, y_test:
        Dados de teste.
    nome:
        Rótulo legível para impressão e gráficos.

    Returns
    -------
    dict com chaves: modelo, acuracia, precisao, recall, f1, auc,
    y_pred, y_prob.
    """
    model.fit(X_train, y_train)

    y_pred: np.ndarray = model.predict(X_test)
    y_prob: np.ndarray = model.predict_proba(X_test)[:, 1]

    metrics: Dict[str, Any] = {
        "modelo":   nome,
        "acuracia": accuracy_score(y_test, y_pred),
        "precisao": precision_score(y_test, y_pred, zero_division=0),
        "recall":   recall_score(y_test, y_pred, zero_division=0),
        "f1":       f1_score(y_test, y_pred, zero_division=0),
        "auc":      roc_auc_score(y_test, y_prob),
        "y_pred":   y_pred,
        "y_prob":   y_prob,
    }

    print(f"\n=== {nome.upper()} ===")
    for k, v in metrics.items():
        if k not in ("modelo", "y_pred", "y_prob"):
            print(f"  {k.capitalize():<10}: {v:.4f}")
    print()
    print(classification_report(y_test, y_pred, target_names=["Mantida", "Cancelada"]))

    return metrics

def _save_figure(fig: plt.Figure, fname: str) -> None:
    """Salva *fig* em OUTPUT_DIR/fname, exibe e fecha."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig.savefig(os.path.join(OUTPUT_DIR, fname), bbox_inches="tight", dpi=120)
    plt.show()
    plt.close(fig)
def plot_model_evaluation(
    metrics: Dict[str, Any],
    y_test,
    cmap: str,
    fname: str,
) -> None:
    """Plota matriz de confusão + curva ROC para um único modelo.

    Parameters
    ----------
    metrics:
        Dicionário retornado por :func:`evaluate_model`.
    y_test:
        Rótulos verdadeiros do conjunto de teste.
    cmap:
        Colormap para a matriz de confusão (ex.: ``"Blues"``).
    fname:
        Nome do arquivo de saída (salvo em OUTPUT_DIR).
    """
    nome: str = metrics["modelo"]
    # Usa o mapeamento estável; fallback para a primeira cor disponível
    cor: str = _MODEL_COLORS.get(nome, next(iter(_MODEL_COLORS.values())))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{nome} — Avaliação", fontsize=14, fontweight="bold")

    # Matriz de confusão
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        metrics["y_pred"],
        display_labels=["Mantida", "Cancelada"],
        cmap=cmap,
        ax=axes[0],
        colorbar=False,
    )
    axes[0].set_title("Matriz de Confusão")

    # Curva ROC
    fpr, tpr, _ = roc_curve(y_test, metrics["y_prob"])
    axes[1].plot(fpr, tpr, color=cor, linewidth=2.5, label=f"AUC = {metrics['auc']:.3f}")
    axes[1].plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.6, label="Aleatório")
    axes[1].fill_between(fpr, tpr, alpha=0.1, color=cor)
    axes[1].set_title("Curva ROC")
    axes[1].set_xlabel("Taxa de Falsos Positivos")
    axes[1].set_ylabel("Taxa de Verdadeiros Positivos")
    axes[1].legend()
    axes[1].set_xlim([0, 1])
    axes[1].set_ylim([0, 1.02])

    plt.tight_layout()
    _save_figure(fig, fname)


def cross_validate_models(
    models: Dict[str, Any],
    X_cv,
    y_cv,
) -> Dict[str, Dict[str, float]]:
    """Stratified K-Fold (k=5) para todos os modelos.

    .. important::
        Recebe estimadores **não treinados** (via :func:`build_models`).
        O scikit-learn re-fita cada clone internamente a cada fold, garantindo
        avaliação sem vazamento de dados.

    Parameters
    ----------
    models:
        Dicionário {nome: estimador não treinado}.
    X_cv, y_cv:
        Features e rótulos para a validação cruzada.

    Returns
    -------
    cv_results : {nome: {acc_mean, acc_std, f1_mean, f1_std, auc_mean, auc_std}}
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_results: Dict[str, Dict[str, float]] = {}
    print("\n=== VALIDAÇÃO CRUZADA ESTRATIFICADA (k=5) ===")

    for nome, modelo in models.items():
        scores = {
            "acc": cross_val_score(modelo, X_cv, y_cv, cv=skf, scoring="accuracy"),
            "f1":  cross_val_score(modelo, X_cv, y_cv, cv=skf, scoring="f1"),
            "auc": cross_val_score(modelo, X_cv, y_cv, cv=skf, scoring="roc_auc"),
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


def plot_kfold(cv_results: Dict[str, Dict[str, float]]) -> None:
    """Barras com desvio padrão para acc, f1 e auc (resultado do K-Fold).

    Parameters
    ----------
    cv_results:
        Dicionário retornado por :func:`cross_validate_models`.
    """
    nomes = list(cv_results.keys())
    # Garante que haja sempre uma cor para cada modelo (cicla se necessário)
    paleta = list(_MODEL_COLORS.values())
    cores = [paleta[i % len(paleta)] for i in range(len(nomes))]

    metrics_info: List[Tuple[str, str, str]] = [
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
        # Ajuste dinâmico do eixo y para evitar clipping das barras de erro
        ax.set_ylim(max(0.0, min(means) - max(stds) - 0.05), 1.05)
        ax.tick_params(axis="x", rotation=15)
        for bar, val, std in zip(bars, means, stds):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + std + 0.005,
                f"{val:.3f}",
                ha="center",
                fontsize=10,
                fontweight="bold",
            )

    plt.tight_layout()
    _save_figure(fig, "11_kfold_comparacao.png")


def plot_roc_comparativo(all_metrics: List[Dict[str, Any]], y_test) -> None:
    """Curvas ROC sobrepostas para todos os modelos.

    Parameters
    ----------
    all_metrics:
        Lista de dicionários retornados por :func:`evaluate_model`.
    y_test:
        Rótulos verdadeiros do conjunto de teste.
    """
    paleta = list(_MODEL_COLORS.values())
    fig, ax = plt.subplots(figsize=(9, 7))

    for i, metrics in enumerate(all_metrics):
        cor = paleta[i % len(paleta)]
        fpr, tpr, _ = roc_curve(y_test, metrics["y_prob"])
        ax.plot(
            fpr, tpr,
            linewidth=2.5,
            color=cor,
            label=f"{metrics['modelo']} (AUC = {metrics['auc']:.3f})",
        )
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
    _save_figure(fig, "12_roc_comparativo.png")


def save_final_model(
    model,
    preprocessor,
    fname: str = "modelo_final.joblib",
) -> None:
    """Salva o melhor classificador e o preprocessor em um único arquivo.

        Nome do arquivo de saída (padrão: ``"modelo_final.joblib"``).
    """
    target_dir = "model"
    os.makedirs(target_dir, exist_ok=True)

    artifacts = {
        "model":        model,
        "preprocessor": preprocessor,
    }

    full_path = os.path.join(target_dir, fname)
    try:
        joblib.dump(artifacts, full_path)
        print(f"\n[SUCESSO] Modelo e Preprocessor salvos em: {full_path}")
    except Exception as exc:  # pragma: no cover
        print(f"\n[ERRO] Falha ao salvar o modelo em '{full_path}': {exc}")
        raise
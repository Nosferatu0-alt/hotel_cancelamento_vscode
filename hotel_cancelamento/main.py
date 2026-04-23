
"""
main.py — Ponto de entrada do projeto Desafio 07
Executa todo o pipeline: EDA → Pré-processamento → Modelagem → Avaliação

Uso:
    python main.py
    python main.py --data caminho/para/hotel_bookings.csv
    python main.py --skip-eda   # pula EDA e vai direto para modelagem
"""

import argparse
import numpy as np

from src.config        import setup_visual
from src.data_loader   import load_dataset, add_temporal_features
from src.eda           import run_eda
from src.preprocessing import (
    prepare_features, split_data, build_preprocessor, fit_transform_data
)
from src.models        import (
    build_models, evaluate_model, plot_model_evaluation,
    cross_validate_models, plot_kfold, plot_roc_comparativo,
)
from src.evaluation    import (
    plot_feature_importance, build_results_table, print_results_table,
    plot_comparacao_final, plot_best_model_importance,
)


MODEL_CMAPS = {
    "Logistic Regression": "Blues",
    "Random Forest":       "Greens",
    "Gradient Boosting":   "Oranges",
}

MODEL_EVAL_FNAMES = {
    "Logistic Regression": "08_lr_avaliacao.png",
    "Random Forest":       "09_rf_avaliacao.png",
    "Gradient Boosting":   "10_gb_avaliacao.png",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Desafio 07 — Cancelamento de Hotel")
    parser.add_argument("--data",     default="data/hotel_bookings.csv",
                        help="Caminho para hotel_bookings.csv")
    parser.add_argument("--skip-eda", action="store_true",
                        help="Pula a etapa de análise exploratória")
    return parser.parse_args()


def main():
    args = parse_args()
    setup_visual()

    print("=" * 60)
    print("DESAFIO 07 — CANCELAMENTO DE RESERVAS DE HOTEL")
    print("=" * 60)

    #  Etapa 0: Carregar e preparar dados 
    print("\n Carregando dataset")
    df = load_dataset(args.data)
    df = add_temporal_features(df)

    #  Etapa 1: EDA 
    if not args.skip_eda:
        run_eda(df)
    else:
        print("\n⏭  EDA ignorada (--skip-eda).")

    #  Etapa 2: Pré-processamento
    print("\n" + "=" * 60)
    print("ETAPA 2 — PRÉ-PROCESSAMENTO")
    print("=" * 60)

    X, y = prepare_features(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    preprocessor = build_preprocessor()
    X_train_proc, X_val_proc, X_test_proc, feature_names = fit_transform_data(
        X_train, X_val, X_test, preprocessor
    )

    # ── Etapa 3: Modelagem ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("ETAPA 3 — COMPARAÇÃO DE CLASSIFICADORES")
    print("=" * 60)

    models     = build_models()
    all_metrics = []

    for nome, model in models.items():
        metrics = evaluate_model(model, X_train_proc, y_train, X_test_proc, y_test, nome)
        all_metrics.append(metrics)
        plot_model_evaluation(
            metrics, y_test,
            cmap=MODEL_CMAPS[nome],
            fname=MODEL_EVAL_FNAMES[nome],
        )

    # Validação cruzada
    X_cv = np.vstack([X_train_proc, X_val_proc])
    y_cv = np.concatenate([y_train, y_val])
    fitted_models = {nome: model for nome, model in models.items()}
    cv_results = cross_validate_models(fitted_models, X_cv, y_cv)
    plot_kfold(cv_results)
    plot_roc_comparativo(all_metrics, y_test)

    # Feature importance (árvores)
    plot_feature_importance(fitted_models, feature_names)

    # ── Etapa 4: Análise comparativa ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("ETAPA 4 — ANÁLISE COMPARATIVA E CONCLUSÕES")
    print("=" * 60)

    df_resultados = build_results_table(all_metrics, cv_results)
    print_results_table(df_resultados)
    plot_comparacao_final(df_resultados)

    # Melhor modelo por F1-Score
    best_nome  = df_resultados.loc[df_resultados["F1-Score"].idxmax(), "Modelo"]
    best_model = fitted_models[best_nome]

    if hasattr(best_model, "feature_importances_"):
        plot_best_model_importance(best_model, best_nome, feature_names)
    else:
        print(f"\n{best_nome} não possui feature_importances_; pulando gráfico de importância final.")

    print("\n✅ Pipeline concluído. Gráficos salvos em outputs/")


if __name__ == "__main__":
    main()

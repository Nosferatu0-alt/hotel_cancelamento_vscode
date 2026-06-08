import streamlit as st
import pandas as pd
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
from src.config import RANDOM_STATE, SAMPLE_SIZE, MONTH_NAME_TO_NUM, SEASON_MAP


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


st.title("Previsão de Cancelamento de Hotel")

with st.expander("Ver Dataframe"):
    df = load_dataset()
    st.dataframe(df)


st.header("Dados da Reserva")
st.caption("Preencha as informações abaixo para prever se a reserva será cancelada.")

tipo_deposito = st.selectbox(
    "Tipo de Depósito",
    ["Selecione...", "Non Refund", "Refundable", "No Deposit"],
    format_func=lambda x: {
        "Selecione...":  "Selecione...",
        "Non Refund":    "Sem reembolso",
        "Refundable":    "Reembolsável",
        "No Deposit":    "Sem depósito",
    }[x],
    help="Se o cliente pagou algum depósito antecipado e se ele pode ser devolvido ou não."
)

antecedencia_reserva = st.number_input(
    "Antecedência da Reserva (dias)",
    min_value=0,
    help="Quantos dias antes do check-in a reserva foi feita."
)

pedidos_extras = st.number_input(
    "Pedidos Extras",
    min_value=0,
    help="Quantidade de pedidos adicionais feitos pelo cliente, como cama extra, berço, quarto no andar mais alto, travesseiro especial, etc."
)

segmento_mercado = st.selectbox(
    "Como a reserva foi feita",
    ["Selecione...", "Online TA", "Offline TA/TO", "Direct", "Groups", "Corporate"],
    format_func=lambda x: {
        "Selecione...":  "Selecione...",
        "Online TA":     "Agência de viagem online (ex: Booking, Expedia)",
        "Offline TA/TO": "Agência de viagem presencial ou operadora de turismo",
        "Direct":        "Direto com o hotel",
        "Groups":        "Reserva em grupo",
        "Corporate":     "Empresa / corporativo",
    }[x],
)

cancelamentos_anteriores = st.number_input(
    "Cancelamentos Anteriores do Cliente",
    min_value=0,
    help="Quantas vezes esse mesmo cliente já cancelou reservas no passado."
)

vagas_estacionamento = st.number_input(
    "Vagas de Estacionamento Necessárias",
    min_value=0,
    help="Número de vagas de estacionamento solicitadas pelo cliente."
)

tarifa_media_diaria = st.number_input(
    "Tarifa Média Diária (R$)",
    min_value=0.0,
    max_value=10000.0,
    step=10.0,
    help="Valor médio cobrado por diária, já considerando descontos ou pacotes aplicados."
)

alteracoes_reserva = st.number_input(
    "Alterações feitas na Reserva",
    min_value=0,
    help="Quantas vezes o cliente modificou a reserva, como trocar a data, o tipo de quarto, etc."
)

tipo_cliente = st.selectbox(
    "Tipo de Cliente",
    ["Selecione...", "Transient", "Transient-Party"],
    format_func=lambda x: {
        "Selecione...":    "Selecione...",
        "Transient":       "Individual (reserva avulsa)",
        "Transient-Party": "Grupo pequeno / acompanhantes",
    }[x],
)


# Treinamento do Modelo
st.divider()
st.subheader("Treinamento do Modelo")
st.caption("Treine o modelo antes de fazer previsões. Isso pode levar alguns instantes.")

if st.button("Treinar Modelo"):
    with st.status("Treinando modelo...", expanded=True) as status:
        st.write("Carregando dados...")
        df = load_dataset()
        df = add_temporal_features(df)

        st.write("Pré-processamento...")
        X, y = prepare_features(df)
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

        preprocessor = build_preprocessor()
        X_train_proc, X_val_proc, X_test_proc, feature_names = fit_transform_data(
            X_train, X_val, X_test, preprocessor
        )

        st.write("Treinando modelos...")
        models = build_models()
        all_metrics = []

        for nome, model in models.items():
            metrics = evaluate_model(model, X_train_proc, y_train, X_test_proc, y_test, nome)
            all_metrics.append(metrics)
            plot_model_evaluation(
                metrics, y_test,
                cmap=MODEL_CMAPS[nome],
                fname=MODEL_EVAL_FNAMES[nome]
            )
            st.session_state[f'modelo_{nome}'] = model
            st.session_state[f'metrics_{nome}'] = metrics

        st.session_state['preprocessor'] = preprocessor
        st.session_state['models'] = models
        status.update(label="Modelo treinado com sucesso!", state="complete")


# Previsão
st.divider()

if st.button("Prever Cancelamento"):

    if 'preprocessor' not in st.session_state or 'models' not in st.session_state:
        st.error("Treine o modelo primeiro antes de fazer previsões. Clique em 'Treinar Modelo' acima.")
        st.stop()

    if tipo_deposito == "Selecione..." or segmento_mercado == "Selecione..." or tipo_cliente == "Selecione...":
        st.warning("Preencha todos os campos do formulário antes de prever.")
        st.stop()

    dados_usuario = pd.DataFrame([{
        # Inputs do usuário
        'deposit_type':                tipo_deposito,
        'lead_time':                   antecedencia_reserva,
        'total_of_special_requests':   pedidos_extras,
        'market_segment':              segmento_mercado,
        'previous_cancellations':      cancelamentos_anteriores,
        'required_car_parking_spaces': vagas_estacionamento,
        'adr':                         tarifa_media_diaria,
        'booking_changes':             alteracoes_reserva,
        'customer_type':               tipo_cliente,

        # Valores padrão
        'stays_in_weekend_nights':         1,
        'stays_in_week_nights':            2,
        'adults':                          2,
        'children':                        0,
        'babies':                          0,
        'previous_bookings_not_canceled':  0,
        'days_in_waiting_list':            0,
        'hotel':                           'City Hotel',
        'meal':                            'BB',
        'distribution_channel':            'TA/TO',
        'reserved_room_type':              'A',
        'estacao':                         'Verão',
    }])

    preprocessor = st.session_state['preprocessor']
    dados_proc = preprocessor.transform(dados_usuario)

    st.subheader("Resultado da Previsão")
    col1, col2, col3 = st.columns(3)

    for col, nome in zip([col1, col2, col3], st.session_state['models'].keys()):
        with col:
            modelo  = st.session_state[f'modelo_{nome}']
            metrics = st.session_state[f'metrics_{nome}']
            pred    = modelo.predict(dados_proc)[0]

            st.markdown(f"**{nome}**")
            if pred == 1:
                st.error("Cancela")
            else:
                st.success("Não Cancela")

            st.markdown("**Metricas do Modelo**")
            st.metric("Acuracia", f"{metrics['acuracia']:.2%}")
            st.metric("Precisao", f"{metrics['precisao']:.2%}")
            st.metric("Recall",   f"{metrics['recall']:.2%}")
            st.metric("F1",       f"{metrics['f1']:.2%}")
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

# titulo
st.title("Previsão de Cancelamento de Hotel")


# Carregamento de dados
if st.button('Treinar modelo'):
    with st.status('Treinando modelo...', expanded = True) as status:
        st.write('Carregando dados...')
        df = load_dataset()
        df = add_temporal_features(df)
        
        st.write('Pré-processamento')
        X, y = prepare_features(df)
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

        preprocessor = build_preprocessor()
        X_train_proc, X_val_proc, X_test_proc, feature_names = fit_transform_data(X_train, X_val, X_test, preprocessor)
        
        st.write('--- Modelagem ---')
        models     = build_models()
        all_metrics = []

        for nome, model in models.items():
            metrics = evaluate_model(model, X_train_proc, y_train, X_test_proc, y_test, nome)
            all_metrics.append(metrics)
            plot_model_evaluation(
                metrics, y_test,
                cmap=MODEL_CMAPS[nome],
                fname=MODEL_EVAL_FNAMES[nome]
        )
        status.update(label="Modelo treinado!", state="complete")
            
        


# previsão

# Coleta de dados do Usuário
col1, col2, col3 = st.columns(3)

with col1:

    deposit_type = st.selectbox('deposit_type (Tipo de depósito):', ['Tipo de depósito','Non Refund','Refundable', 'No Deposit'])

    lead_time = st.number_input('lead_time (Antecedência da reserva):', min_value=0)

    total_of_special_requests = st.number_input('total_of_special_requests (Total de solicitações especiais):', min_value= 0)

    market_segment = st.selectbox('market_segment (Segmento de mercado):',['Segmento de mercado','Online TA','Offline TA/TO', 'Direct', 'Groups', 'Corporate'])

    previous_cancellations = st.number_input('previous_cancellations (Cancelamentos anteriores):', min_value= 0)
    
    required_car_parking_spaces = st.number_input('required_car_parking_spaces (Vagas de estacionamento necessárias):', min_value= 0)

    adr = st.number_input('Tarifa Média Diária (ADR):', min_value= 0.0)

    booking_changes = st.number_input('booking_changes (Alterações na reserva):', min_value= 0)

    customer_type = st.selectbox('customer_type (Tipo de cliente):',['Tipo de cliente','Transient', 'Transient-Party'])

with col2:
    
    with st.expander('Ver Dataframe'):
        df = load_dataset()
        st.dataframe(df)


with col3:
    st.text('Gráficos')
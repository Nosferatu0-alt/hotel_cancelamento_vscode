import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from src.config        import setup_visual
from src.data_loader   import load_dataset, add_temporal_features
from src.eda           import run_eda
from src.preprocessing import (
    prepare_features, split_data, build_preprocessor, fit_transform_data
)
from model.models import (
    build_models, evaluate_model, plot_model_evaluation,
    cross_validate_models, plot_kfold, plot_roc_comparativo,
    save_final_model
)
from src.evaluation import (
    plot_feature_importance, build_results_table, print_results_table,
    plot_comparacao_final, plot_best_model_importance,
)
from src.config import RANDOM_STATE, SAMPLE_SIZE, MONTH_NAME_TO_NUM, SEASON_MAP


# ── Configuração da página ──────────────────────────────────────────────────
st.set_page_config(
    page_title="HotelIQ — Previsão de Cancelamento",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Design system ───────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=DM+Serif+Display:ital@0;1&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background-color: #0F1117;
    color: #E8EAF0;
}

header[data-testid="stHeader"] { background: transparent; }

/* ── Tabs ── */
div[data-testid="stTabs"] > div:first-child {
    border-bottom: 1px solid #1E2535 !important;
    gap: 4px;
}
button[data-baseweb="tab"] {
    font-size: 13px !important;
    font-weight: 500 !important;
    color: #4B5563 !important;
    background: transparent !important;
    border-radius: 6px 6px 0 0 !important;
    padding: 10px 18px !important;
    border: none !important;
    transition: color 0.15s ease !important;
}
button[data-baseweb="tab"]:hover {
    color: #9CA3AF !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #818CF8 !important;
    border-bottom: 2px solid #6366F1 !important;
}
div[data-testid="stTabsContent"] {
    padding-top: 28px !important;
}

/* ── Hero ── */
.hero {
    background: linear-gradient(135deg, #1A1F2E 0%, #0F1117 60%, #1A1F2E 100%);
    border-bottom: 1px solid #2A2F3F;
    padding: 48px 40px 36px;
    margin: -1rem -1rem 0 -1rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: "";
    position: absolute;
    top: -60px; right: -80px;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(99,102,241,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.hero-eyebrow {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #6366F1;
    margin-bottom: 12px;
}
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 38px;
    font-weight: 400;
    line-height: 1.15;
    color: #F0F1F6;
    margin: 0 0 10px;
}
.hero-title em { font-style: italic; color: #818CF8; }
.hero-sub {
    font-size: 15px;
    color: #6B7280;
    font-weight: 400;
    max-width: 520px;
}

/* ── Status badge ── */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    background: #1E2535;
    border: 1px solid #2E3548;
    border-radius: 100px;
    padding: 6px 14px;
    font-size: 12px;
    font-weight: 500;
    color: #9CA3AF;
    margin-top: 20px;
}
.status-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: #10B981;
    box-shadow: 0 0 6px #10B981;
    flex-shrink: 0;
}
.status-dot.offline { background: #6B7280; box-shadow: none; }

/* ── Section label ── */
.section-label {
    font-size: 10.5px;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #4B5563;
    margin: 40px 0 20px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.section-label::after {
    content: "";
    flex: 1;
    height: 1px;
    background: #1E2535;
}

/* ── Inputs ── */
div[data-testid="stSelectbox"] > div > div,
div[data-testid="stNumberInput"] > div > div > input {
    background-color: #1A1F2E !important;
    border: 1px solid #2A2F3F !important;
    border-radius: 8px !important;
    color: #E8EAF0 !important;
    font-size: 14px !important;
}
div[data-testid="stSelectbox"] > div > div:hover,
div[data-testid="stNumberInput"] > div > div > input:focus {
    border-color: #6366F1 !important;
    box-shadow: 0 0 0 2px rgba(99,102,241,0.15) !important;
}
label[data-testid="stWidgetLabel"] p {
    font-size: 13px !important;
    font-weight: 500 !important;
    color: #9CA3AF !important;
}

/* ── Botões ── */
div[data-testid="stButton"] > button {
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    letter-spacing: 0.02em !important;
    transition: all 0.18s ease !important;
    border: none !important;
    padding: 10px 22px !important;
    width: 100%;
}
div[data-testid="stButton"]:first-of-type > button {
    background: #1E2535 !important;
    color: #9CA3AF !important;
    border: 1px solid #2A2F3F !important;
}
div[data-testid="stButton"]:first-of-type > button:hover {
    background: #252C40 !important;
    color: #E8EAF0 !important;
    border-color: #4B5563 !important;
}
.primary-btn > button {
    background: linear-gradient(135deg, #6366F1, #818CF8) !important;
    color: #fff !important;
    box-shadow: 0 4px 20px rgba(99,102,241,0.3) !important;
}
.primary-btn > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 28px rgba(99,102,241,0.4) !important;
}

/* ── Resultado ── */
.result-cancel {
    background: linear-gradient(135deg, rgba(239,68,68,0.1), rgba(239,68,68,0.04));
    border: 1px solid rgba(239,68,68,0.3);
    border-left: 4px solid #EF4444;
    border-radius: 12px;
    padding: 22px 24px;
}
.result-ok {
    background: linear-gradient(135deg, rgba(16,185,129,0.1), rgba(16,185,129,0.04));
    border: 1px solid rgba(16,185,129,0.3);
    border-left: 4px solid #10B981;
    border-radius: 12px;
    padding: 22px 24px;
}
.result-title {
    font-family: 'DM Serif Display', serif;
    font-size: 22px;
    font-weight: 400;
    margin: 4px 0 6px;
    color: #F0F1F6;
}
.result-desc { font-size: 13px; color: #6B7280; line-height: 1.5; }

/* ── Métricas ── */
.metric-card {
    background: #161B27;
    border: 1px solid #1E2535;
    border-radius: 10px;
    padding: 16px 18px;
    text-align: center;
}
.metric-value {
    font-family: 'DM Serif Display', serif;
    font-size: 26px;
    color: #818CF8;
    line-height: 1;
    margin-bottom: 4px;
}
.metric-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #4B5563;
}

/* ── Model chip ── */
.model-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #1E2535;
    border: 1px solid #2A2F3F;
    border-radius: 6px;
    padding: 4px 10px;
    font-size: 12px;
    font-weight: 600;
    color: #9CA3AF;
    margin-bottom: 10px;
}

/* ── Comparativo ── */
.compare-cancel {
    background: rgba(239,68,68,0.08);
    border: 1px solid rgba(239,68,68,0.2);
    border-radius: 10px;
    padding: 14px 16px;
    text-align: center;
    font-weight: 600;
    font-size: 13px;
    color: #F87171;
}
.compare-ok {
    background: rgba(16,185,129,0.08);
    border: 1px solid rgba(16,185,129,0.2);
    border-radius: 10px;
    padding: 14px 16px;
    text-align: center;
    font-weight: 600;
    font-size: 13px;
    color: #34D399;
}

/* ── Gráficos placeholder ── */
.chart-placeholder {
    background: #161B27;
    border: 1px dashed #2A2F3F;
    border-radius: 12px;
    padding: 48px 24px;
    text-align: center;
    color: #374151;
    font-size: 13px;
}

div[data-testid="stAlert"] {
    border-radius: 10px !important;
    border: 1px solid #2A2F3F !important;
    background: #161B27 !important;
}
details summary { font-size: 13px !important; font-weight: 500 !important; color: #6B7280 !important; }
details summary:hover { color: #9CA3AF !important; }
hr { border-color: #1E2535 !important; margin: 32px 0 !important; }
div[data-testid="stStatus"] {
    background: #161B27 !important;
    border: 1px solid #2A2F3F !important;
    border-radius: 10px !important;
}

/* Matplotlib / pyplot fundo transparente */
div[data-testid="stImage"] img {
    border-radius: 10px;
    border: 1px solid #1E2535;
}
</style>
""", unsafe_allow_html=True)


# ── Constantes ──────────────────────────────────────────────────────────────
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
CAMINHO_MODELO = os.path.join("model", "modelo_final.joblib")
REPORTS_DIR    = os.path.join(BASE_DIR, "reports")


# ── Carrega modelo salvo ─────────────────────────────────────────────────────
if 'modelo_carregado' not in st.session_state:
    if os.path.exists(CAMINHO_MODELO):
        try:
            artefatos = joblib.load(CAMINHO_MODELO)
            st.session_state['modelo_final']     = artefatos["model"]
            st.session_state['preprocessor']     = artefatos["preprocessor"]
            st.session_state['modelo_carregado'] = True
        except Exception:
            st.session_state['modelo_carregado'] = False
    else:
        st.session_state['modelo_carregado'] = False


# ── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">Hotel Intelligence · Modelo Preditivo</div>
    <h1 class="hero-title">Previsão de<br><em>Cancelamento</em></h1>
    <p class="hero-sub">Insira os dados da reserva e obtenha uma análise instantânea de risco com nosso modelo Gradient Boosting.</p>
    {badge}
</div>
""".format(
    badge='<div class="status-badge"><span class="status-dot"></span>Modelo pronto para uso</div>'
    if st.session_state['modelo_carregado']
    else '<div class="status-badge"><span class="status-dot offline"></span>Sem modelo treinado — treine abaixo</div>'
), unsafe_allow_html=True)


# ── Dataset expandível ───────────────────────────────────────────────────────
with st.expander("📋  Visualizar dataset completo"):
    df = load_dataset()
    st.dataframe(df, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# ABAS PRINCIPAIS
# ════════════════════════════════════════════════════════════════════════════
tab_previsao, tab_graficos = st.tabs([
    "🔍  Previsão",
    "📊  Gráficos & Avaliação",
])


# ╔══════════════════════════════════════════════════════════════╗
# ║  ABA 1 — PREVISÃO                                           ║
# ╚══════════════════════════════════════════════════════════════╝
with tab_previsao:

    # ── Formulário ──────────────────────────────────────────────
    st.markdown('<div class="section-label">Dados da Reserva</div>', unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns(3, gap="medium")

    with col_a:
        tipo_deposito = st.selectbox(
            "Tipo de Depósito",
            ["Selecione...", "Non Refund", "Refundable", "No Deposit"],
            format_func=lambda x: {
                "Selecione...": "Selecione...",
                "Non Refund":   "Sem reembolso",
                "Refundable":   "Reembolsável",
                "No Deposit":   "Sem depósito",
            }[x],
            help="Se o cliente pagou algum depósito antecipado e se ele pode ser devolvido ou não.",
        )
        antecedencia_reserva = st.number_input(
            "Antecedência da Reserva (dias)", min_value=0,
            help="Quantos dias antes do check-in a reserva foi feita.",
        )
        pedidos_extras = st.number_input(
            "Pedidos Extras", min_value=0,
            help="Cama extra, berço, andar especial, travesseiro etc.",
        )

    with col_b:
        segmento_mercado = st.selectbox(
            "Canal de Reserva",
            ["Selecione...", "Online TA", "Offline TA/TO", "Direct", "Groups", "Corporate"],
            format_func=lambda x: {
                "Selecione...":  "Selecione...",
                "Online TA":     "Agência online (Booking, Expedia…)",
                "Offline TA/TO": "Agência presencial / operadora",
                "Direct":        "Direto com o hotel",
                "Groups":        "Reserva em grupo",
                "Corporate":     "Empresa / corporativo",
            }[x],
        )
        cancelamentos_anteriores = st.number_input(
            "Cancelamentos Anteriores", min_value=0,
            help="Quantas vezes esse cliente já cancelou no passado.",
        )
        vagas_estacionamento = st.number_input("Vagas de Estacionamento", min_value=0)

    with col_c:
        tarifa_media_diaria = st.number_input(
            "Tarifa Média Diária (R$)", min_value=0.0, max_value=10000.0, step=10.0,
            help="Valor médio por diária considerando descontos.",
        )
        alteracoes_reserva = st.number_input(
            "Alterações na Reserva", min_value=0,
            help="Quantas vezes o cliente modificou datas, quarto etc.",
        )
        tipo_cliente = st.selectbox(
            "Perfil do Cliente",
            ["Selecione...", "Transient", "Transient-Party"],
            format_func=lambda x: {
                "Selecione...":    "Selecione...",
                "Transient":       "Individual (reserva avulsa)",
                "Transient-Party": "Grupo pequeno / acompanhantes",
            }[x],
        )

    # ── Treinamento ─────────────────────────────────────────────
    st.markdown('<div class="section-label">Modelo</div>', unsafe_allow_html=True)

    btn_col, _ = st.columns([1, 3])
    with btn_col:
        treinar = st.button("⚙️  Treinar Modelo", use_container_width=True)

    if treinar:
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

            st.write("Treinando classificadores...")
            models = build_models()
            all_metrics = []

            for nome, model in models.items():
                metrics = evaluate_model(model, X_train_proc, y_train, X_test_proc, y_test, nome)
                all_metrics.append(metrics)
                plot_model_evaluation(
                    metrics, y_test,
                    cmap=MODEL_CMAPS[nome],
                    fname=MODEL_EVAL_FNAMES[nome],
                )
                st.session_state[f'modelo_{nome}']  = model
                st.session_state[f'metrics_{nome}'] = metrics

            st.session_state['preprocessor']     = preprocessor
            st.session_state['models']            = models
            st.session_state['X_test_proc']       = X_test_proc
            st.session_state['y_test']            = y_test
            st.session_state['feature_names']     = feature_names

            st.write("Exportando modelo final...")
            save_final_model(models["Gradient Boosting"], preprocessor)

            st.session_state['modelo_final']     = models["Gradient Boosting"]
            st.session_state['modelo_carregado'] = True

            status.update(label="Modelo treinado e salvo com sucesso! Veja os gráficos na aba 📊", state="complete")

    # ── Previsão ─────────────────────────────────────────────────
    st.markdown('<div class="section-label">Previsão</div>', unsafe_allow_html=True)

    prev_col, _ = st.columns([1, 3])
    with prev_col:
        st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
        prever = st.button("🔍  Prever Cancelamento", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if prever:
        if not st.session_state['modelo_carregado'] and (
            'preprocessor' not in st.session_state or 'models' not in st.session_state
        ):
            st.error("Treine o modelo antes de fazer previsões.")
            st.stop()

        if tipo_deposito == "Selecione..." or segmento_mercado == "Selecione..." or tipo_cliente == "Selecione...":
            st.warning("Preencha todos os campos antes de prever.")
            st.stop()

        dados_usuario = pd.DataFrame([{
            'deposit_type':                tipo_deposito,
            'lead_time':                   antecedencia_reserva,
            'total_of_special_requests':   pedidos_extras,
            'market_segment':              segmento_mercado,
            'previous_cancellations':      cancelamentos_anteriores,
            'required_car_parking_spaces': vagas_estacionamento,
            'adr':                         tarifa_media_diaria,
            'booking_changes':             alteracoes_reserva,
            'customer_type':               tipo_cliente,
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
        dados_proc   = preprocessor.transform(dados_usuario)

        if 'modelo_final' in st.session_state:
            modelo = st.session_state['modelo_final']
            pred   = modelo.predict(dados_proc)[0]

            st.markdown('<div style="margin-top:8px;">', unsafe_allow_html=True)
            if pred == 1:
                st.markdown("""
                <div class="result-cancel">
                    <div style="font-size:28px;margin-bottom:4px;">⚠️</div>
                    <div class="result-title">Alta probabilidade de cancelamento</div>
                    <div class="result-desc">O modelo indica risco elevado. Considere acionar protocolos de retenção ou confirmar a reserva com o cliente.</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="result-ok">
                    <div style="font-size:28px;margin-bottom:4px;">✅</div>
                    <div class="result-title">Reserva com baixo risco de cancelamento</div>
                    <div class="result-desc">O modelo indica que esta reserva tem boa probabilidade de se concretizar. Nenhuma ação adicional necessária.</div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Comparativo dos 3 modelos
        if 'models' in st.session_state:
            st.markdown('<div class="section-label" style="margin-top:32px;">Comparativo de Classificadores</div>', unsafe_allow_html=True)

            cols = st.columns(3, gap="medium")
            for col, nome in zip(cols, st.session_state['models'].keys()):
                with col:
                    m     = st.session_state[f'modelo_{nome}']
                    metr  = st.session_state[f'metrics_{nome}']
                    p_ind = m.predict(dados_proc)[0]

                    st.markdown(f'<div class="model-chip">◆ {nome}</div>', unsafe_allow_html=True)
                    if p_ind == 1:
                        st.markdown('<div class="compare-cancel">⚠ Cancela</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="compare-ok">✓ Não Cancela</div>', unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)
                    m1, m2 = st.columns(2)
                    with m1:
                        st.markdown(f'<div class="metric-card"><div class="metric-value">{metr["acuracia"]:.0%}</div><div class="metric-label">Acurácia</div></div>', unsafe_allow_html=True)
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.markdown(f'<div class="metric-card"><div class="metric-value">{metr["recall"]:.0%}</div><div class="metric-label">Recall</div></div>', unsafe_allow_html=True)
                    with m2:
                        st.markdown(f'<div class="metric-card"><div class="metric-value">{metr["precisao"]:.0%}</div><div class="metric-label">Precisão</div></div>', unsafe_allow_html=True)
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.markdown(f'<div class="metric-card"><div class="metric-value">{metr["f1"]:.0%}</div><div class="metric-label">F1-Score</div></div>', unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════╗
# ║  ABA 2 — GRÁFICOS & AVALIAÇÃO                               ║
# ╚══════════════════════════════════════════════════════════════╝
with tab_graficos:

    if not st.session_state.get('modelo_carregado') and 'models' not in st.session_state:
        st.markdown("""
        <div class="chart-placeholder">
            <div style="font-size:36px;margin-bottom:12px;">📊</div>
            <div style="font-size:14px;color:#6B7280;">Treine o modelo primeiro para visualizar os gráficos de avaliação.</div>
        </div>
        """, unsafe_allow_html=True)

    else:
        # ── Sub-abas por modelo ──────────────────────────────────
        st.markdown('<div class="section-label">Avaliação por Modelo</div>', unsafe_allow_html=True)

        sub_lr, sub_rf, sub_gb = st.tabs([
            "📘 Logistic Regression",
            "📗 Random Forest",
            "📙 Gradient Boosting",
        ])

        def render_model_charts(sub_tab, nome, fname):
            with sub_tab:
                img_path = os.path.join(REPORTS_DIR, fname)
                if os.path.exists(img_path):
                    st.image(img_path, use_container_width=True)
                else:
                    st.markdown(f'<div class="chart-placeholder">Gráfico <code>{fname}</code> não encontrado em reports/.</div>', unsafe_allow_html=True)

        render_model_charts(sub_lr, "Logistic Regression", MODEL_EVAL_FNAMES["Logistic Regression"])
        render_model_charts(sub_rf, "Random Forest",       MODEL_EVAL_FNAMES["Random Forest"])
        render_model_charts(sub_gb, "Gradient Boosting",   MODEL_EVAL_FNAMES["Gradient Boosting"])

        # ── Gráficos globais ─────────────────────────────────────
        st.markdown('<div class="section-label" style="margin-top:40px;">Comparativo Geral</div>', unsafe_allow_html=True)

        global_charts = {
            "ROC Comparativo":         "11_roc_comparativo.png",
            "Comparação Final":        "12_comparacao_final.png",
            "Feature Importance (GB)": "13_best_model_importance.png",
        }

        g_cols = st.columns(len(global_charts), gap="medium")
        for col, (label, fname) in zip(g_cols, global_charts.items()):
            with col:
                img_path = os.path.join(REPORTS_DIR, fname)
                st.caption(label)
                if os.path.exists(img_path):
                    st.image(img_path, use_container_width=True)
                else:
                    st.markdown(f'<div class="chart-placeholder" style="padding:32px 16px;">Gráfico não gerado ainda.<br><small>{fname}</small></div>', unsafe_allow_html=True)

        # ── K-Fold ───────────────────────────────────────────────
        kfold_path = os.path.join(REPORTS_DIR, "14_kfold.png")
        if os.path.exists(kfold_path):
            st.markdown('<div class="section-label" style="margin-top:40px;">Validação Cruzada (K-Fold)</div>', unsafe_allow_html=True)
            st.image(kfold_path, use_container_width=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:64px;padding:24px 0;border-top:1px solid #1E2535;
     text-align:center;font-size:12px;color:#374151;letter-spacing:0.04em;">
    HotelIQ · Modelo de Machine Learning para Previsão de Cancelamentos
</div>
""", unsafe_allow_html=True)
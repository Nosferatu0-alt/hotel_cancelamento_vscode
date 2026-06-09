import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from io import BytesIO

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

from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, classification_report
)


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

CAMINHO_MODELO = os.path.join(BASE_DIR, "model", "modelo_final.joblib")


# ── Helpers de gráficos inline ───────────────────────────────────────────────
def _fig_to_bytes(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor="#161B27", edgecolor="none")
    buf.seek(0)
    plt.close(fig)
    return buf


def plot_confusion_matrix_inline(model, X_test, y_test, cmap, nome):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3.5))
    fig.patch.set_facecolor("#161B27")
    ax.set_facecolor("#161B27")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Mantida", "Cancelada"])
    disp.plot(ax=ax, cmap=cmap, values_format="d")  # FIX: era colormap=, agora cmap=
    ax.set_title(f"Matriz de Confusão — {nome}", color="#E8EAF0", fontsize=11, pad=10)
    for text in ax.texts:
        text.set_color("#E8EAF0")
    ax.tick_params(colors="#9CA3AF")
    ax.xaxis.label.set_color("#9CA3AF")
    ax.yaxis.label.set_color("#9CA3AF")
    plt.tight_layout()
    return _fig_to_bytes(fig)


def plot_roc_inline(model, X_test, y_test, nome, color):
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(4, 3.5))
    fig.patch.set_facecolor("#161B27")
    ax.set_facecolor("#161B27")
    ax.plot(fpr, tpr, color=color, lw=2, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], color="#374151", linestyle="--", lw=1)
    ax.set_xlabel("Taxa de Falsos Positivos", color="#9CA3AF", fontsize=10)
    ax.set_ylabel("Taxa de Verdadeiros Positivos", color="#9CA3AF", fontsize=10)
    ax.set_title(f"Curva ROC — {nome}", color="#E8EAF0", fontsize=11, pad=10)
    ax.legend(facecolor="#1E2535", labelcolor="#E8EAF0", fontsize=10)
    ax.tick_params(colors="#9CA3AF")
    for spine in ax.spines.values():
        spine.set_edgecolor("#2A2F3F")
    plt.tight_layout()
    return _fig_to_bytes(fig)


def plot_roc_comparativo_inline(models, X_test, y_test):
    colors = {"Logistic Regression": "#818CF8", "Random Forest": "#34D399", "Gradient Boosting": "#FBBF24"}
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor("#161B27")
    ax.set_facecolor("#161B27")
    for nome, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors[nome], lw=2, label=f"{nome} (AUC={roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="#374151", linestyle="--", lw=1)
    ax.set_xlabel("FPR", color="#9CA3AF", fontsize=10)
    ax.set_ylabel("TPR", color="#9CA3AF", fontsize=10)
    ax.set_title("ROC Comparativo", color="#E8EAF0", fontsize=11, pad=10)
    ax.legend(facecolor="#1E2535", labelcolor="#E8EAF0", fontsize=9)
    ax.tick_params(colors="#9CA3AF")
    for spine in ax.spines.values():
        spine.set_edgecolor("#2A2F3F")
    plt.tight_layout()
    return _fig_to_bytes(fig)


def plot_comparacao_final_inline(all_metrics):
    nomes     = [m["nome"]     for m in all_metrics]
    acuracias = [m["acuracia"] for m in all_metrics]
    f1s       = [m["f1"]       for m in all_metrics]
    aucs      = [m["auc"]      for m in all_metrics]

    x = np.arange(len(nomes))
    w = 0.25
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor("#161B27")
    ax.set_facecolor("#161B27")
    ax.bar(x - w, acuracias, w, label="Acurácia", color="#818CF8", alpha=0.9)
    ax.bar(x,     f1s,       w, label="F1-Score", color="#34D399", alpha=0.9)
    ax.bar(x + w, aucs,      w, label="AUC",      color="#FBBF24", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(nomes, color="#9CA3AF", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_title("Comparação Final dos Modelos", color="#E8EAF0", fontsize=11, pad=10)
    ax.legend(facecolor="#1E2535", labelcolor="#E8EAF0", fontsize=9)
    ax.tick_params(colors="#9CA3AF")
    for spine in ax.spines.values():
        spine.set_edgecolor("#2A2F3F")
    plt.tight_layout()
    return _fig_to_bytes(fig)


def plot_feature_importance_inline(model, feature_names):
    if not hasattr(model, "feature_importances_"):
        return None
    importances = model.feature_importances_
    indices = np.argsort(importances)[-15:]
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor("#161B27")
    ax.set_facecolor("#161B27")
    ax.barh(
        [feature_names[i] for i in indices],
        importances[indices],
        color="#FBBF24", alpha=0.85
    )
    ax.set_title("Feature Importance — Gradient Boosting", color="#E8EAF0", fontsize=11, pad=10)
    ax.tick_params(colors="#9CA3AF", labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2A2F3F")
    plt.tight_layout()
    return _fig_to_bytes(fig)


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
        )
        antecedencia_reserva = st.number_input("Antecedência da Reserva (dias)", min_value=0)
        pedidos_extras = st.number_input("Pedidos Extras", min_value=0)

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
        cancelamentos_anteriores = st.number_input("Cancelamentos Anteriores", min_value=0)
        vagas_estacionamento = st.number_input("Vagas de Estacionamento", min_value=0)

    with col_c:
        tarifa_media_diaria = st.number_input("Tarifa Média Diária (R$)", min_value=0.0, max_value=10000.0, step=10.0)
        alteracoes_reserva = st.number_input("Alterações na Reserva", min_value=0)
        tipo_cliente = st.selectbox(
            "Perfil do Cliente",
            ["Selecione...", "Transient", "Transient-Party"],
            format_func=lambda x: {
                "Selecione...":    "Selecione...",
                "Transient":       "Individual (reserva avulsa)",
                "Transient-Party": "Grupo pequeno / acompanhantes",
            }[x],
        )

    # ── Treinamento ──────────────────────────────────────────────
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
                metrics["nome"] = nome  # FIX: garante que a chave "nome" existe
                all_metrics.append(metrics)
                # FIX: salva cada modelo e métricas individualmente no session_state
                st.session_state[f'modelo_{nome}']  = model
                st.session_state[f'metrics_{nome}'] = metrics

            st.session_state['preprocessor']  = preprocessor
            st.session_state['models']         = models
            st.session_state['X_test_proc']    = X_test_proc
            st.session_state['y_test']         = y_test
            st.session_state['feature_names']  = feature_names
            st.session_state['all_metrics']    = all_metrics

            st.write("Exportando modelo final...")
            save_final_model(models["Gradient Boosting"], preprocessor)

            st.session_state['modelo_final']     = models["Gradient Boosting"]
            st.session_state['modelo_carregado'] = True

            status.update(label="Modelo treinado! Veja os gráficos na aba 📊", state="complete")

    # ── Previsão ──────────────────────────────────────────────────
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

        # FIX: só mostra o comparativo se os modelos individuais foram treinados nessa sessão
        modelos_individuais_prontos = (
            'models' in st.session_state and
            all(f'metrics_{n}' in st.session_state for n in st.session_state['models'].keys())
        )

        if modelos_individuais_prontos:
            st.markdown('<div class="section-label" style="margin-top:32px;">Comparativo de Classificadores</div>', unsafe_allow_html=True)

            cols = st.columns(3, gap="medium")
            for col, nome in zip(cols, st.session_state['models'].keys()):
                with col:
                    m     = st.session_state['models'][nome]
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

    modelos_prontos = 'models' in st.session_state and 'X_test_proc' in st.session_state

    if not modelos_prontos:
        st.markdown("""
        <div class="chart-placeholder">
            <div style="font-size:36px;margin-bottom:12px;">📊</div>
            <div style="font-size:14px;color:#6B7280;">Treine o modelo primeiro para visualizar os gráficos de avaliação.</div>
        </div>
        """, unsafe_allow_html=True)

    else:
        models        = st.session_state['models']
        X_test_proc   = st.session_state['X_test_proc']
        y_test        = st.session_state['y_test']
        feature_names = st.session_state.get('feature_names', [])
        all_metrics   = st.session_state.get('all_metrics', [])

        MODEL_COLORS = {
            "Logistic Regression": "#818CF8",
            "Random Forest":       "#34D399",
            "Gradient Boosting":   "#FBBF24",
        }

        # ── Sub-abas por modelo ──────────────────────────────────
        st.markdown('<div class="section-label">Avaliação por Modelo</div>', unsafe_allow_html=True)

        sub_lr, sub_rf, sub_gb = st.tabs([
            "📘 Logistic Regression",
            "📗 Random Forest",
            "📙 Gradient Boosting",
        ])

        for sub_tab, nome in zip([sub_lr, sub_rf, sub_gb], models.keys()):
            with sub_tab:
                model = models[nome]
                cmap  = MODEL_CMAPS[nome]
                color = MODEL_COLORS[nome]

                c1, c2 = st.columns(2, gap="medium")
                with c1:
                    st.caption("Matriz de Confusão")
                    buf = plot_confusion_matrix_inline(model, X_test_proc, y_test, cmap, nome)
                    st.image(buf, use_container_width=True)
                with c2:
                    st.caption("Curva ROC")
                    buf = plot_roc_inline(model, X_test_proc, y_test, nome, color)
                    st.image(buf, use_container_width=True)

                metr = st.session_state.get(f'metrics_{nome}')
                if metr:
                    st.markdown("<br>", unsafe_allow_html=True)
                    mc1, mc2, mc3, mc4 = st.columns(4, gap="small")
                    for col, (label, key) in zip(
                        [mc1, mc2, mc3, mc4],
                        [("Acurácia","acuracia"),("Precisão","precisao"),("Recall","recall"),("F1-Score","f1")]
                    ):
                        with col:
                            st.markdown(
                                f'<div class="metric-card"><div class="metric-value">{metr[key]:.0%}</div>'
                                f'<div class="metric-label">{label}</div></div>',
                                unsafe_allow_html=True
                            )

        # ── Gráficos globais ─────────────────────────────────────
        st.markdown('<div class="section-label" style="margin-top:40px;">Comparativo Geral</div>', unsafe_allow_html=True)

        g1, g2, g3 = st.columns(3, gap="medium")

        with g1:
            st.caption("ROC Comparativo")
            buf = plot_roc_comparativo_inline(models, X_test_proc, y_test)
            st.image(buf, use_container_width=True)

        with g2:
            st.caption("Comparação Final")
            if all_metrics:
                buf = plot_comparacao_final_inline(all_metrics)
                st.image(buf, use_container_width=True)
            else:
                st.markdown('<div class="chart-placeholder" style="padding:32px 16px;">Dados indisponíveis.</div>', unsafe_allow_html=True)

        with g3:
            st.caption("Feature Importance (GB)")
            if feature_names:
                buf = plot_feature_importance_inline(models["Gradient Boosting"], feature_names)
                if buf:
                    st.image(buf, use_container_width=True)
                else:
                    st.markdown('<div class="chart-placeholder" style="padding:32px 16px;">Modelo sem feature_importances_.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="chart-placeholder" style="padding:32px 16px;">Feature names indisponíveis.</div>', unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:64px;padding:24px 0;border-top:1px solid #1E2535;
     text-align:center;font-size:12px;color:#374151;letter-spacing:0.04em;">
    HotelIQ · Modelo de Machine Learning para Previsão de Cancelamentos
</div>
""", unsafe_allow_html=True)
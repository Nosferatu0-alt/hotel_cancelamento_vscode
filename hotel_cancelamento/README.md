#  Desafio 07 — Cancelamento de Reservas de Hotel

**Disciplina:** IA - Classificadores  
**Grupo:** 7 | **Domínio:** Hotelaria  
**Tipo:** Classificação Binária  
**Dataset:** [jessemostipak/hotel-booking-demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)

---

##  Objetivo

Prever se uma reserva de hotel será **cancelada** (`is_canceled = 1`) ou **mantida** (`is_canceled = 0`).

Cancelamentos de última hora causam prejuízos significativos. Prever quais reservas têm maior probabilidade de cancelamento permite:
- Overbooking estratégico
- Ações de retenção proativas
- Otimização de receita

---

 Estrutura do Projeto

```
hotel_cancelamento/
│
├── main.py                  # Ponto de entrada — executa o pipeline completo
│
├── src/                     # Módulos do projeto
│   ├── __init__.py
│   ├── config.py            # Constantes globais, features, visual
│   ├── data_loader.py       # Carregamento e amostragem do dataset
│   ├── eda.py               # Análise Exploratória (EDA) — 7 gráficos
│   ├── preprocessing.py     # Encoding, split e pipeline sklearn
│   ├── models.py            # Treinamento, avaliação e curvas ROC
│   └── evaluation.py        # Feature Importance e tabela comparativa
│
├── data/
│   └── hotel_bookings.csv   # ← coloque o dataset aqui
│
├── outputs/                 # Gráficos gerados automaticamente
│
├── requirements.txt
└── README.md
```

---

##  Como rodar

### 1. Clonar / abrir no VSCode

```bash
# Abrir a pasta no VSCode
code hotel_cancelamento/
```

### 2. Criar ambiente virtual (recomendado)

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate
```

### 3. Instalar dependências

```bash
pip install -r requirements.txt
```

### 4. Baixar o dataset

**Via Kaggle CLI:**
```bash
kaggle datasets download -d jessemostipak/hotel-booking-demand
unzip hotel-booking-demand.zip -d data/
```

**Ou manualmente:**  
Acesse https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand, baixe `hotel_bookings.csv` e coloque em `data/`.

### 5. Executar

```bash
# Pipeline completo (EDA + modelagem)
python main.py

# Especificar caminho do CSV
python main.py --data caminho/para/hotel_bookings.csv

# Pular EDA e ir direto para modelagem
python main.py --skip-eda
```

---

##  Pipeline de Execução

| Etapa | Módulo | Descrição |
|-------|--------|-----------|
| **0** | `data_loader.py` | Carrega CSV, amostragem estratificada (n=2.000) |
| **1** | `eda.py` | Análise Exploratória — 7 gráficos |
| **2** | `preprocessing.py` | Encoding OHE + StandardScaler + split 70/15/15 |
| **3** | `models.py` | Treina LR, RF, GB; K-Fold; curvas ROC |
| **4** | `evaluation.py` | Feature Importance + tabela comparativa final |

---

##  Módulos

### `src/config.py`
Centraliza todas as constantes do projeto:
- `RANDOM_STATE`, `SAMPLE_SIZE`
- `FEATURES_NUM`, `FEATURES_CAT`, `TARGET`
- Mapeamentos de meses/estações
- Paleta de cores e configuração visual

### `src/data_loader.py`
- `load_dataset(path)` — lê CSV e retorna amostra estratificada
- `add_temporal_features(df)` — adiciona `arrival_month_name` e `estacao`

### `src/eda.py`
Gera e salva em `outputs/`:
| Arquivo | Conteúdo |
|---------|----------|
| `01_taxa_cancelamento.png` | Barra + pizza da distribuição geral |
| `02_cancelamento_hotel.png` | Volume e taxa por tipo de hotel |
| `03_lead_time.png` | Distribuição, boxplot e faixas de lead time |
| `04_correlacoes.png` | Top 15 correlações + heatmap |
| `05_categoricas.png` | Cancelamento por depósito, cliente, canal, refeição |
| `06_missing_outliers.png` | Nulos e outliers em lead_time |
| `07_sazonalidade.png` | Cancelamento por mês e estação |

### `src/preprocessing.py`
- `build_preprocessor()` — ColumnTransformer (mediana + OHE)
- `prepare_features(df)` — seleciona e trata nulos básicos
- `split_data(X, y)` — divisão estratificada 70/15/15
- `fit_transform_data(...)` — fit no treino, transform nos três splits

### `src/models.py`
- `build_models()` — LogisticRegression, RandomForest, GradientBoosting
- `evaluate_model(...)` — treina e retorna métricas + predições
- `plot_model_evaluation(...)` — matriz de confusão + ROC por modelo
- `cross_validate_models(...)` — Stratified K-Fold k=5
- `plot_kfold(...)` — barras com desvio padrão
- `plot_roc_comparativo(...)` — curvas ROC sobrepostas

### `src/evaluation.py`
- `plot_feature_importance(...)` — Top 20 para RF e GB
- `build_results_table(...)` — DataFrame com todas as métricas
- `print_results_table(...)` — tabela formatada + melhor modelo
- `plot_comparacao_final(...)` — barras + radar chart
- `plot_best_model_importance(...)` — importância e curva acumulada

---

##  Modelos Avaliados

| Modelo | Descrição |
|--------|-----------|
| **Logistic Regression** | Baseline linear; interpretável e rápido |
| **Random Forest** | Ensemble de árvores; captura não-linearidades |
| **Gradient Boosting** | Boosting sequencial; geralmente melhor desempenho |

**Métricas reportadas:** Acurácia, Precisão, Recall, F1-Score, AUC-ROC, AUC CV (k=5)

---

##  Features Utilizadas

**Numéricas (13):** `lead_time`, `stays_in_weekend_nights`, `stays_in_week_nights`, `adults`, `children`, `babies`, `previous_cancellations`, `previous_bookings_not_canceled`, `booking_changes`, `adr`, `total_of_special_requests`, `days_in_waiting_list`, `required_car_parking_spaces`

**Categóricas (8):** `hotel`, `meal`, `market_segment`, `distribution_channel`, `reserved_room_type`, `deposit_type`, `customer_type`, `estacao`

---

##  Conclusões

O **Gradient Boosting** tende a superar os demais por:
- Capacidade não-linear e robustez a outliers
- Combinação sequencial de árvores fracas em modelo forte

**Features mais relevantes:**
- `lead_time` — mais antecipada a reserva, maior risco
- `deposit_type` — depósito não reembolsável reduz cancelamentos
- `previous_cancellations` — histórico é forte preditor
- `total_of_special_requests` — pedidos especiais indicam engajamento
- `adr` — diárias mais altas correlacionam com padrões específicos

> **Para produção:** retreinar com o dataset completo (119.390 registros), aplicar hiperparametrização (GridSearchCV/Optuna) e monitorar drift do modelo.

---

##  Requisitos

- Python 3.9+
- Ver `requirements.txt`

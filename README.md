![Python](https://img.shields.io/badge/python-3.9+-blue.svg) ![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white) ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?logo=pandas&logoColor=white) ![Status](https://img.shields.io/badge/status-concluído-brightgreen)

# Previsão de Cancelamento de Reservas em Hotéis

Machine Learning aplicado ao setor de hotelaria

##  Integrantes

| Nome | RA |
|---|---|
| Lucia Maria Reis Braga | 2035292 |
| Kenji Yuri Mitsuka de Paula | 2033472 |
| Matheus Bargas Rodrigues Flausino | 2057008 |

---

##  Visão Geral

O problema é tratado como classificação binária:
- `1` → Reserva cancelada
- `0` → Reserva mantida

---

##  Pipeline do Projeto

```text
Dados → Limpeza → EDA → Pré-processamento → Modelagem → Avaliação
```

---

##  Descrição do Problema

O setor hoteleiro sofre prejuízos significativos com cancelamentos de reservas de última hora. Quando um cliente cancela sem aviso prévio, o hotel perde receita e dificilmente consegue preencher o quarto no mesmo período.

---

##  Objetivo do Projeto

Desenvolver um modelo de Machine Learning capaz de prever se uma reserva de hotel será **cancelada ou mantida**, com base nas características da reserva e do cliente, e disponibilizar esse modelo em uma aplicação web interativa via Streamlit.

---

##  Dataset Utilizado

- **Nome:** Hotel Booking Demand
- **Arquivo:** `hotel_bookings.csv`
- **Fonte:** Kaggle — [Hotel Booking Demand Dataset](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)
- **Total de registros:** ~119.000 reservas
- **Período:** 2015 a 2017
- **Variável alvo:** `is_canceled` (0 = Mantida, 1 = Cancelada)

---

##  Tipo de Problema de Machine Learning

**Classificação Binária Supervisionada** — prever se uma reserva será cancelada (1) ou mantida (0).

---
# Metodologia
 
1. **Carregamento e amostragem** — amostra estratificada de 10.000 registros (~8,4% do dataset completo) para balancear as classes
2. **Análise Exploratória (EDA)** — distribuições, correlações, sazonalidade e outliers
3. **Pré-processamento** — encoding de variáveis categóricas, normalização de numéricas e engenharia de features temporais (estação do ano)
4. **Divisão dos dados** — treino (70%) / validação (15%) / teste (15%)
5. **Treinamento e avaliação** — três classificadores comparados com métricas múltiplas
6. **Validação Cruzada** — Stratified K-Fold (k=5) aplicado sobre treino + validação (8.500 amostras) para avaliação robusta da estabilidade dos modelos
7. **Exportação** — modelo final salvo em `modelo_final.joblib`
8. **Deploy** — aplicação interativa publicada no Streamlit Cloud

---

##  Modelos Treinados e tabela comparativa

Foram treinados e comparados três classificadores:

- **Logistic Regression** — modelo linear como baseline
- **Random Forest** — ensemble de árvores de decisão
- **Gradient Boosting** — boosting sequencial de árvores
-
- ### Tabela Comparativa de Modelos

| Modelo | Acurácia | Precisão | Recall | F1-Score | AUC-ROC | AUC CV (média) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.7913 | 0.8015 | 0.5809 | 0.6736 | 0.8334 | 0.8493 |
| **Random Forest** | 0.7967 | 0.8910 | 0.5144 | 0.6522 | 0.8588 | 0.8744 |
| **Gradient Boosting** | 0.8080 | 0.8160 | 0.6223 | 0.7061 | 0.8661 | 0.8793 |

---

##  Modelo Final Escolhido 

**Gradient Boosting** foi selecionado como modelo final por apresentar o melhor equilíbrio entre acurácia, F1-Score e AUC-ROC.

Hiperparâmetros utilizados:
- `n_estimators = 100`
- `learning_rate = 0.1`
- `max_depth = 4`

---

## Métricas de Avaliação

### Comparativo entre modelos

| Modelo | Acurácia | Precisão | Recall | F1-Score | AUC-ROC |
|---|---|---|---|---|---|
| Logistic Regression | 79,13% | 80,15% | 58,09% | 67,36% | 0,8334 |
| Random Forest | 79,67% | 89,10% | 51,44% | 65,22% | 0,8588 |
| **Gradient Boosting** | **80,80%** | **81,60%** | **62,23%** | **70,61%** | **86,61** |


### Tabela Comparativa de Modelos

| Modelo | Acurácia | Precisão | Recall | F1-Score | AUC-ROC | AUC CV (média) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.7913 | 0.8015 | 0.5809 | 0.6736 | 0.8334 | 0.8493 |
| **Random Forest** | 0.7967 | 0.8910 | 0.5144 | 0.6522 | 0.8588 | 0.8744 |
| **Gradient Boosting** | 0.8080 | 0.8160 | 0.6223 | 0.7061 | 0.8661 | 0.8793 |
---

##  Principais Resultados

- O **Gradient Boosting** obteve a melhor AUC-ROC (0,866), indicando boa capacidade de separação entre as classes
- As features mais relevantes para a previsão foram: tipo de depósito, antecedência da reserva, segmento de mercado e número de cancelamentos anteriores
- O modelo identifica corretamente 92% das reservas mantidas e 62% das canceladas
- A validação cruzada (K-Fold k=5) confirmou a estabilidade do modelo

## Feature Importance
 
Principais variáveis que influenciam o cancelamento (importância do Gradient Boosting):
 
- **deposit_type_Non Refund** — feature mais importante, respondendo por 46,34% da importância total do modelo
- **lead_time** — tempo entre a reserva e o check-in (12,35%)
- **total_of_special_requests** — clientes com mais pedidos especiais cancelam menos (7,82%)
- **market_segment_Online TA** — reservas via agências online têm maior propensão a cancelamento (7,27%)
- **previous_cancellations** — histórico de cancelamentos anteriores do cliente (6,14%)
---


## Estrutura dos Arquivos
![Python](https://img.shields.io/badge/python-3.9+-blue.svg) ![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white) ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?logo=pandas&logoColor=white) ![Status](https://img.shields.io/badge/status-concluído-brightgreen)

# Previsão de Cancelamento de Reservas em Hotéis

Machine Learning aplicado ao setor de hotelaria

## Integrantes

| Nome | RA |
|---|---|
| Lucia Maria Reis Braga | 2035292 |
| Kenji Yuri Mitsuka de Paula | 2033472 |
| Matheus Bargas Rodrigues Flausino | 2057008 |

---

## Visão Geral

O setor hoteleiro sofre prejuízos significativos com cancelamentos de reservas de última hora. Quando um cliente cancela sem aviso prévio, o hotel perde receita e dificilmente consegue preencher o quarto no mesmo período. Identificar com antecedência quais reservas têm alta probabilidade de cancelamento permite que os gestores tomem ações preventivas, como políticas de overbooking controlado ou contato proativo com o cliente.

O problema é tratado como classificação binária:
- `1` → Reserva cancelada
- `0` → Reserva mantida

---

## Pipeline do Projeto

```text
Dados → Limpeza → EDA → Pré-processamento → Modelagem → Avaliação
```

---

## Descrição do Problema

O setor hoteleiro sofre prejuízos significativos com cancelamentos de reservas de última hora. Quando um cliente cancela sem aviso prévio, o hotel perde receita e dificilmente consegue preencher o quarto no mesmo período.

---

## Objetivo do Projeto

Desenvolver um modelo de Machine Learning capaz de prever se uma reserva de hotel será **cancelada ou mantida**, com base nas características da reserva e do cliente, e disponibilizar esse modelo em uma aplicação web interativa via Streamlit.

---

## Dataset Utilizado

- **Nome:** Hotel Booking Demand
- **Arquivo:** `hotel_bookings.csv`
- **Fonte:** Kaggle — [Hotel Booking Demand Dataset](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)
- **Total de registros:** ~119.000 reservas
- **Período:** 2015 a 2017
- **Variável alvo:** `is_canceled` (0 = Mantida, 1 = Cancelada)

---

## Tipo de Problema de Machine Learning

**Classificação Binária Supervisionada** — prever se uma reserva será cancelada (1) ou mantida (0).

---

## Metodologia

1. **Carregamento e amostragem** — amostra estratificada de 10.000 registros (~8,4% do dataset completo) para balancear as classes
2. **Análise Exploratória (EDA)** — distribuições, correlações, sazonalidade e outliers
3. **Pré-processamento** — encoding de variáveis categóricas, normalização de numéricas e engenharia de features temporais (estação do ano)
4. **Divisão dos dados** — treino (70%) / validação (15%) / teste (15%)
5. **Treinamento e avaliação** — três classificadores comparados com métricas múltiplas
6. **Validação Cruzada** — Stratified K-Fold (k=5) aplicado sobre treino + validação (8.500 amostras) para avaliação robusta da estabilidade dos modelos
7. **Exportação** — modelo final salvo em `modelo_final.joblib`
8. **Deploy** — aplicação interativa publicada no Streamlit Cloud

---

## Modelos Treinados

Foram treinados e comparados três classificadores:

- **Logistic Regression** — modelo linear como baseline
- **Random Forest** — ensemble de árvores de decisão (100 estimadores, max_depth=8)
- **Gradient Boosting** — boosting sequencial de árvores

---

## Modelo Final Escolhido

**Gradient Boosting** foi selecionado como modelo final por apresentar o melhor equilíbrio entre acurácia, F1-Score e AUC-ROC.

Hiperparâmetros utilizados:
- `n_estimators = 100`
- `learning_rate = 0.1`
- `max_depth = 4`
---

## Principais Resultados

- O **Gradient Boosting** obteve a melhor AUC-ROC (0,8661), indicando boa capacidade de separação entre as classes
- A feature mais relevante para a previsão foi o **deposit_type Non Refund** (46,34% de importância), seguida por **lead_time** (12,35%), **total_of_special_requests** (7,82%), **market_segment_Online TA** (7,27%) e **previous_cancellations** (6,14%)
- O modelo identifica corretamente 92% das reservas mantidas e 62% das canceladas
- A validação cruzada (K-Fold k=5) confirmou a estabilidade do modelo

## Feature Importance

Principais variáveis que influenciam o cancelamento (importância do Gradient Boosting):

- **Tipo de depósito Non Refund** — feature mais importante, com 46,34% da importância total
- **Lead time** — tempo entre a reserva e o check-in (12,35%)
- **Total de pedidos especiais** — clientes com mais pedidos cancelam menos (7,82%)
- **Segmento de mercado Online TA** — canal pelo qual a reserva foi feita (7,27%)
- **Histórico de cancelamentos** — cancelamentos anteriores do cliente (6,14%)

---

## Estrutura dos Arquivos
```hotel_cancelamento_vscode/

├── hotel_cancelamento/
│   ├── app.py                    # Aplicação Streamlit
│   ├── main.py                   # Script de execução local
│   ├── requirements.txt          # Dependências do projeto
│   ├── data/
│   │   └── hotel_bookings.csv    # Dataset
│   ├── model/
│   │   ├── __init__.py
│   │   ├── models.py             # Treinamento e avaliação dos modelos
│   │   └── modelo_final.joblib   # Modelo exportado
│   ├── src/
│   │   ├── __init__.py
│   │   ├── config.py             # Configurações e constantes
│   │   ├── data_loader.py        # Carregamento do dataset
│   │   ├── eda.py                # Análise exploratória
│   │   ├── preprocessing.py      # Pré-processamento
│   │   └── evaluation.py        # Avaliação e visualizações
│   ├── notebooks/
│   │   └── rascunho_hotel_cancelamento.ipynb
│   └── reports/
│       └── Relatório_atualizado.PDF
└── requirements.txt              # Dependências (raiz para Streamlit Cloud)
``
---

## Tecnologias Utilizadas

| Tecnologia | Versão | Uso |
|---|---|---|
| Python | 3.x | Linguagem principal |
| Streamlit | — | Interface web interativa |
| Scikit-learn | — | Modelos de ML e pré-processamento |
| Pandas | — | Manipulação de dados |
| NumPy | — | Operações numéricas |
| Matplotlib | — | Visualizações |
| Seaborn | — | Visualizações estatísticas |
| Joblib | — | Serialização do modelo |

---

## Instruções para Executar o Notebook

```bash
# 1. Clone o repositório
git clone https://github.com/Nosferatu0-alt/hotel_cancelamento_vscode
cd hotel_cancelamento_vscode/hotel_cancelamento

# 2. Crie e ative um ambiente virtual
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
.venv\Scripts\activate           # Windows

# 3. Instale as dependências
pip install -r requirements.txt

# 4. Abra o Jupyter
jupyter notebook notebooks/rascunho_hotel_cancelamento.ipynb
```

---

## Instruções para Executar o App Streamlit

```bash
# 1. Clone o repositório
git clone https://github.com/Nosferatu0-alt/hotel_cancelamento_vscode
cd hotel_cancelamento_vscode/hotel_cancelamento

# 2. Instale as dependências
pip install -r requirements.txt

# 3. Execute o app
streamlit run app.py
```

O app abrirá automaticamente em `http://localhost:8501`

---

## Link do App Publicado
(https://hotelcancelamentovscode-zp3jmgbh9jhnpwpw2daj8v.streamlit.app)

---

## Limitações

- O modelo foi treinado com dados de hotéis europeus (2015–2017), podendo não generalizar bem para outros contextos geográficos ou períodos
- A amostragem de 10.000 registros representa cerca de 8,4% do dataset completo (~119.000 registros), o que pode reduzir a capacidade do modelo de capturar padrões menos frequentes
- O Recall para a classe "Cancelada" (62%) indica que cerca de 38% dos cancelamentos não são detectados pelo modelo
- Variáveis como comportamento pós-reserva e histórico completo do cliente não estão disponíveis no dataset
- A taxa de cancelamento de 99,4% associada ao `deposit_type Non Refund` é um padrão atípico que merece investigação adicional, pois pode refletir um viés nos dados originais

---

## Conclusão

O **Gradient Boosting** se destacou como o modelo mais adequado, atingindo **80,8% de acurácia** e **AUC-ROC de 0,866**, mostrando boa capacidade de discriminação entre reservas mantidas e canceladas.

A aplicação desenvolvida em Streamlit permite que qualquer usuário, sem conhecimento técnico, preencha os dados de uma reserva e obtenha uma previsão em tempo real, tornando o modelo acessível e utilizável na prática.

---

## Melhorias Futuras

- Otimização de hiperparâmetros com GridSearchCV ou Optuna
- Teste com modelos mais robustos (XGBoost, LightGBM)
- Aumentar o tamanho da amostra de treinamento
- Adicionar explicabilidade com SHAP values
- Investigar o comportamento atípico do `deposit_type Non Refund` nos dados originais

---

## Feedback

Este projeto faz parte do aprendizado em Machine Learning.
Sugestões são bem-vindas, sinta-se à vontade para abrir uma issue ou contribuir.

---

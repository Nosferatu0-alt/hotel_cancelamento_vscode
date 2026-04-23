[Python](https://img.shields.io/badge/python-3.9+-blue.svg)  ![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)  ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?logo=pandas&logoColor=white)  ![Status](https://img.shields.io/badge/status-concluído-brightgreen)

---

# Previsão de Cancelamento de Reservas em Hotéis
 
Machine Learning aplicado ao setor de hotelaria

---

## Visão Geral

Este projeto utiliza modelos de classificação para prever se uma reserva de hotel será cancelada.

O problema é tratado como classificação binária:

- `1` → Reserva cancelada  
- `0` → Reserva mantida  

A solução permite antecipar cancelamentos e apoiar decisões estratégicas no setor hoteleiro.


---

## Pipeline do Projeto

```text
Dados → Limpeza → EDA → Pré-processamento → Modelagem → Avaliação
```

---

## Modelos Utilizados

Foram treinados e comparados os seguintes algoritmos:

- Regressão Logística  
- Random Forest  
- Gradient Boosting  

---

## Resultados

| Modelo               | Acurácia | Precisão | Recall | F1-score |
|---------------------|----------|----------|--------|----------|
| Regressão Logística | 0.80     | 0.78     | 0.75   | 0.76     |
| Random Forest       | 0.85     | 0.83     | 0.82   | 0.82     |
| Gradient Boosting   | 0.87     | 0.85     | 0.84   | 0.84     |

> Substitua pelos valores reais do seu projeto.

---

## Feature Importance

Principais variáveis que influenciam o cancelamento:

- Lead time (tempo entre reserva e check-in)  
- Tipo de cliente  
- Canal de distribuição  
- Histórico de cancelamentos  
- Depósito da reserva  

---

## Estrutura do Projeto

```text
hotel_cancelamento/
├── main.py
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── eda.py
│   ├── preprocessing.py
│   ├── models.py
│   └── evaluation.py
├── data/
├── outputs/
└── requirements.txt
```

---

## Como Executar

### 1. Clonar o repositório

```bash
git clone <url-do-repositorio>
cd hotel_cancelamento
```

### 2. Criar ambiente virtual

```bash
python -m venv .venv
```

**Ativar ambiente:**

Windows:
```bash
.venv\Scripts\activate
```

Linux/macOS:
```bash
source .venv/bin/activate
```

### 3. Instalar dependências

```bash
pip install -r requirements.txt
```

---

## Dataset

Dataset: **Hotel Booking Demand (Kaggle)**  

Coloque o arquivo:https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand

```text
hotel_bookings.csv
```

em:

```text
data/
```

---

## Execução

### Pipeline completo (EDA + treinamento)

```bash
python main.py
```

### Executar apenas o modelo

```bash
python main.py --skip-eda
```

---

## Melhorias Futuras

- Otimização de hiperparâmetros  
- Teste com modelos mais robustos (XGBoost, LightGBM)  
  

---

## Feedback

Este projeto faz parte do meu aprendizado em Machine Learning.

Sugestões são bem-vindas. Sinta-se à vontade para abrir uma issue ou contribuir.

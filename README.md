[Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?logo=pandas&logoColor=white)
![Status](https://img.shields.io/badge/status-concluído-brightgreen)

> **Desafio 07 | Grupo 7** -> Classificadores  -> Domínio: Hotelaria

##  Objetivo do Projeto

Este projeto utiliza técnicas de Machine Learning para prever se uma reserva de hotel será **cancelada** (`is_canceled = 1`) ou **mantida** (`is_canceled = 0`). 

A capacidade de prever cancelamentos permite que gestores hoteleiros tomem decisões baseadas em dados, como:
* Redução de prejuízos por cancelamentos de última hora.
*  Realização de **overbooking estratégico**.
*  Ações proativas de retenção de clientes.

---

##  Estrutura do Repositório

O projeto foi modularizado para garantir organização e facilidade de manutenção:

```text
hotel_cancelamento/
├── main.py              # Ponto de entrada (executa o pipeline completo)
├── src/                 # Módulos do projeto
│   ├── config.py        # Configurações globais e constantes
│   ├── data_loader.py   # Carregamento e tratamento temporal
│   ├── eda.py           # Análise Exploratória (7 gráficos automáticos)
│   ├── preprocessing.py # Encoding e normalização (Sklearn Pipeline)
│   ├── models.py        # Treinamento e avaliação (LR, RF, GB)
│   └── evaluation.py    # Comparação de métricas e Feature Importance
├── data/                # Local para o dataset (hotel_bookings.csv)
├── outputs/             # Gráficos gerados automaticamente
└── requirements.txt     # Dependências do sistema
```
**como executar?**

# Clone o repositório e acesse a pasta
git clone <url-do-seu-repositorio>
cd hotel_cancelamento

# Crie e ative um ambiente virtual
python -m venv .venv
# Windows: .venv\Scripts\activate | Linux/macOS: source .venv/bin/activate

# Instale as dependências
pip install -r requirements.txt

O modelo utiliza o dataset Hotel Booking Demand (Kaggle).

Certifique-se de colocar o arquivo hotel_bookings.csv dentro da pasta data/

# Executar análise completa (EDA + Treinamento)
python main.py

# Pular a análise gráfica e ir direto para o modelo
python main.py --skip-eda


---
## Feedback

Este é o meu primeiro treino de Machine Learning e estou em constante aprendizado! 

Se você tiver qualquer sugestão de melhoria no código, na análise de dados ou na escolha dos modelos, **estou totalmente aberta a qualquer feedback que puder me dar.** Sinta-se à vontade para abrir uma *Issue* ou entrar em contato.
---

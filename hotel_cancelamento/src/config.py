
"""
config.py — Configurações globais do projeto
"""

import warnings
import matplotlib.pyplot as plt
import seaborn as sns

# Seed de reprodutibilidade
RANDOM_STATE = 42

# Tamanho da amostra
SAMPLE_SIZE = 2000

#  Splits 
TEST_SIZE       = 0.30   # 30 % → val+test
VAL_FROM_TEMP   = 0.50   # metade do temp → validação

#  Features 
FEATURES_NUM = [
    "lead_time",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "booking_changes",
    "adr",
    "total_of_special_requests",
    "days_in_waiting_list",
    "required_car_parking_spaces",
]

FEATURES_CAT = [
    "hotel",
    "meal",
    "market_segment",
    "distribution_channel",
    "reserved_room_type",
    "deposit_type",
    "customer_type",
    "estacao",
]

TARGET = "is_canceled"

#  Mapeamentos de datas 
MONTH_NAME_TO_NUM = {
    "January": 1, "February": 2, "March": 3,    "April": 4,
    "May": 5,     "June": 6,     "July": 7,     "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12,
}

SEASON_MAP = {
    1: "Verão", 2: "Verão",   3: "Outono",    4: "Outono",
    5: "Outono", 6: "Inverno", 7: "Inverno",   8: "Inverno",
    9: "Primavera", 10: "Primavera", 11: "Primavera", 12: "Verão",
}

#Saída de plots 
OUTPUT_DIR = "outputs"

#  Visual 
COLORS = {
    "green":  "#2ecc71",
    "red":    "#e74c3c",
    "blue":   "#3498db",
    "orange": "#e67e22",
    "purple": "#9b59b6",
}


def setup_visual() -> None:
    """Aplica tema global de visualização."""
    warnings.filterwarnings("ignore")
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)
    plt.rcParams["figure.dpi"]    = 110
    plt.rcParams["axes.titlesize"] = 13
    plt.rcParams["axes.labelsize"] = 11

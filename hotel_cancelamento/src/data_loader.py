
"""
data_loader.py — Carregamento e amostragem do dataset
"""

import pandas as pd
from src.config import RANDOM_STATE, SAMPLE_SIZE, MONTH_NAME_TO_NUM, SEASON_MAP


def load_dataset(path: str = "data/hotel_bookings.csv") -> pd.DataFrame:
    """
    Carrega o CSV completo e retorna amostra estratificada de SAMPLE_SIZE linhas.

    Parameters
    ----------
    path : str
        Caminho para o arquivo hotel_bookings.csv

    Returns
    -------
    pd.DataFrame
        Amostra com SAMPLE_SIZE linhas, distribuição de `is_canceled` preservada.
    """
    df_full = pd.read_csv(path)
    print(f"Dataset completo: {df_full.shape[0]:,} linhas × {df_full.shape[1]} colunas")
    print(f"Distribuição alvo (completo):\n{df_full['is_canceled'].value_counts()}\n")

    # Amostragem estratificada
    df = (
        df_full.groupby("is_canceled", group_keys=False)
        .apply(lambda x: x.sample(frac=SAMPLE_SIZE / len(df_full), random_state=RANDOM_STATE))
        .reset_index(drop=True)
    )
    df = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE).reset_index(drop=True)

    print(f"Amostra: {df.shape[0]:,} linhas × {df.shape[1]} colunas")
    print(f"Taxa de cancelamento: {df['is_canceled'].mean():.2%}\n")
    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adiciona colunas de mês numérico e estação do ano.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame com coluna `arrival_date_month`

    Returns
    -------
    pd.DataFrame
        DataFrame com colunas `arrival_month_name` (int) e `estacao` (str).
    """
    df = df.copy()
    df["arrival_month_name"] = df["arrival_date_month"].map(MONTH_NAME_TO_NUM)
    df["estacao"] = df["arrival_month_name"].map(SEASON_MAP)
    return df

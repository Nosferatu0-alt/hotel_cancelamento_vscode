
"""
preprocessing.py — Pré-processamento, encoding e divisão de dados
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from typing import Tuple

from src.config import (
    FEATURES_NUM, FEATURES_CAT, TARGET,
    RANDOM_STATE, TEST_SIZE, VAL_FROM_TEMP,
)


def build_preprocessor() -> ColumnTransformer:
    """
    Cria o ColumnTransformer com pipelines para numéricas e categóricas.

    Returns
    -------
    ColumnTransformer
        Preprocessador não ajustado.
    """
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe",     OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", num_pipeline, FEATURES_NUM),
        ("cat", cat_pipeline, FEATURES_CAT),
    ])


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Seleciona features e trata nulos básicos antes do pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame com colunas brutas (incluindo `estacao`).

    Returns
    -------
    X : pd.DataFrame
    y : pd.Series
    """
    features_all = FEATURES_NUM + FEATURES_CAT
    df_model = df[features_all + [TARGET]].copy()

    for col in FEATURES_NUM:
        if df_model[col].isnull().sum() > 0:
            df_model[col].fillna(df_model[col].median(), inplace=True)
    for col in FEATURES_CAT:
        if df_model[col].isnull().sum() > 0:
            df_model[col].fillna(df_model[col].mode()[0], inplace=True)

    print(f"Dataset para modelagem: {df_model.shape}")
    print(f"Nulos restantes: {df_model.isnull().sum().sum()}")
    return df_model.drop(TARGET, axis=1), df_model[TARGET]


def split_data(X: pd.DataFrame, y: pd.Series):
    """
    Divisão estratificada 70 % treino / 15 % validação / 15 % teste.

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test
    """
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=VAL_FROM_TEMP, stratify=y_temp, random_state=RANDOM_STATE
    )

    print(f"\nDivisão do dataset:")
    print(f"  Treino:    {X_train.shape[0]:,} ({X_train.shape[0]/len(X)*100:.1f}%)")
    print(f"  Validação: {X_val.shape[0]:,} ({X_val.shape[0]/len(X)*100:.1f}%)")
    print(f"  Teste:     {X_test.shape[0]:,} ({X_test.shape[0]/len(X)*100:.1f}%)")
    print(f"  Taxa cancelamento — Treino: {y_train.mean():.2%} | Val: {y_val.mean():.2%} | Teste: {y_test.mean():.2%}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def fit_transform_data(X_train, X_val, X_test, preprocessor: ColumnTransformer):
    """
    Ajusta o preprocessador no treino e transforma os três splits.

    Returns
    -------
    X_train_proc, X_val_proc, X_test_proc, feature_names
    """
    X_train_proc = preprocessor.fit_transform(X_train)
    X_val_proc   = preprocessor.transform(X_val)
    X_test_proc  = preprocessor.transform(X_test)

    ohe_names = preprocessor.named_transformers_["cat"]["ohe"].get_feature_names_out(FEATURES_CAT)
    feature_names = FEATURES_NUM + list(ohe_names)

    print(f"\nShape pós-encoding:")
    print(f"  Treino:    {X_train_proc.shape}")
    print(f"  Validação: {X_val_proc.shape}")
    print(f"  Teste:     {X_test_proc.shape}")
    print(f"  Features totais: {X_train_proc.shape[1]}")
    return X_train_proc, X_val_proc, X_test_proc, feature_names

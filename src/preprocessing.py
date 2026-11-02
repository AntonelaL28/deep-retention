"""Carga, división y preprocesamiento de los datos de clientes.

Para evitar fuga de datos (data leakage), el preprocesador se ajusta
únicamente con el conjunto de entrenamiento y luego se aplica tal cual
a validación, prueba y a los clientes nuevos de la aplicación.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config

PREPROCESSOR_FILE = config.MODELS_DIR / "preprocessor.joblib"


def load_data(path: str | Path = config.RAW_DATA_FILE) -> pd.DataFrame:
    """Lee el CSV de clientes."""
    return pd.read_csv(path)


def get_feature_columns(df: pd.DataFrame) -> dict[str, list[str]]:
    """Devuelve las variables (numéricas, binarias y categóricas) presentes en df.

    Se excluyen identificadores, la variable objetivo y las columnas con fuga.
    """
    def present(cols):
        return [c for c in cols if c in df.columns]

    return {
        "numeric": present(config.NUMERIC_FEATURES),
        "binary": present(config.BINARY_FEATURES),
        "categorical": present(config.CATEGORICAL_FEATURES),
    }


def split_data(df: pd.DataFrame,
               target: str = config.TARGET,
               test_size: float = config.TEST_SIZE,
               val_size: float = config.VALIDATION_SIZE,
               random_state: int = config.RANDOM_STATE):
    """Divide en entrenamiento, validación y prueba de forma estratificada.

    La estratificación mantiene la misma proporción de clientes que abandonan
    en los tres conjuntos, importante porque las clases están desbalanceadas.
    Devuelve (X_train, X_val, X_test, y_train, y_val, y_test).
    """
    if target not in df.columns:
        raise ValueError(
            f"El dataset no contiene la variable objetivo '{target}'. "
            "No es posible entrenar un modelo supervisado sin ella.")

    X = df.drop(columns=[target])
    y = df[target].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state)

    # val_size se expresa sobre el total, se ajusta respecto a lo que queda
    val_rel = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_rel, stratify=y_train,
        random_state=random_state)

    return X_train, X_val, X_test, y_train, y_val, y_test


def build_preprocessor(columns: dict[str, list[str]]) -> ColumnTransformer:
    """Crea el transformador: StandardScaler para numéricas, OneHotEncoder
    para categóricas y las binarias (0/1) sin cambios.

    Cualquier otra columna (identificadores, columnas con fuga) se descarta.
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), columns["numeric"]),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             columns["categorical"]),
            ("bin", "passthrough", columns["binary"]),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def fit_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    """Ajusta el preprocesador SOLO con el conjunto de entrenamiento."""
    preprocessor = build_preprocessor(get_feature_columns(X_train))
    preprocessor.fit(X_train)
    return preprocessor


def save_preprocessor(preprocessor: ColumnTransformer,
                      path: str | Path = PREPROCESSOR_FILE) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, path)


def load_preprocessor(path: str | Path = PREPROCESSOR_FILE) -> ColumnTransformer:
    return joblib.load(path)

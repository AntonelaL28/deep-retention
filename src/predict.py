"""Inferencia: carga el modelo entrenado y estima la probabilidad de fuga.

Separado del entrenamiento; lo usa la aplicación.
"""
from __future__ import annotations

from functools import lru_cache

import pandas as pd
from tensorflow import keras

from src import config, preprocessing


@lru_cache(maxsize=1)
def load_artifacts():
    """Carga (una sola vez) el modelo y el preprocesador guardados."""
    model = keras.models.load_model(config.MODEL_FILE)
    prep = preprocessing.load_preprocessor()
    return model, prep


def predict_proba(df: pd.DataFrame) -> pd.Series:
    """Probabilidad de fuga (0-1) para cada cliente de df."""
    model, prep = load_artifacts()
    probs = model.predict(prep.transform(df), verbose=0).ravel()
    return pd.Series(probs, index=df.index, name="prob_churn")

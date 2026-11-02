"""Inferencia: carga el modelo entrenado y estima la probabilidad de fuga.

Separado del entrenamiento; lo usa la aplicación.
"""
from __future__ import annotations

from functools import lru_cache

import pandas as pd
from tensorflow import keras

from src import calibration, config, preprocessing


@lru_cache(maxsize=1)
def load_artifacts():
    """Carga (una sola vez) el modelo y el preprocesador guardados."""
    model = keras.models.load_model(config.MODEL_FILE)
    prep = preprocessing.load_preprocessor()
    return model, prep


@lru_cache(maxsize=1)
def _load_calibrator():
    if calibration.CALIBRATOR_FILE.exists():
        return calibration.load_calibrator()
    return None


def predict_raw(df: pd.DataFrame) -> pd.Series:
    """Salida directa de la red (sin calibrar)."""
    model, prep = load_artifacts()
    probs = model.predict(prep.transform(df), verbose=0).ravel()
    return pd.Series(probs, index=df.index, name="prob_raw")


def predict_proba(df: pd.DataFrame) -> pd.Series:
    """Probabilidad de fuga calibrada (0-1) para cada cliente de df."""
    raw = predict_raw(df)
    cal = _load_calibrator()
    probs = cal.transform(raw.to_numpy()) if cal is not None else raw.to_numpy()
    return pd.Series(probs, index=df.index, name="prob_churn")

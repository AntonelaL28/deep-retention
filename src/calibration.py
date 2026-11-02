"""Calibración de probabilidades (Platt scaling).

class_weight hace que la red sobreestime la probabilidad de fuga. Como la
fórmula financiera usa esa probabilidad directamente, se corrige con una
regresión logística sobre el logit de la red, ajustada con validación.

Uso:  python -m src.calibration
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from src import config

CALIBRATOR_FILE = config.MODELS_DIR / "calibrator.joblib"


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


class PlattCalibrator:
    def fit(self, raw_probs, y):
        self.lr_ = LogisticRegression().fit(_logit(raw_probs), y)
        return self

    def transform(self, raw_probs):
        return self.lr_.predict_proba(_logit(raw_probs))[:, 1]


def expected_calibration_error(y, probs, n_bins: int = 10) -> float:
    """ECE: diferencia media entre probabilidad predicha y tasa real por tramo."""
    y, probs = np.asarray(y), np.asarray(probs)
    bins = np.minimum((probs * n_bins).astype(int), n_bins - 1)
    return float(sum(abs(probs[bins == b].mean() - y[bins == b].mean()) * (bins == b).mean()
                     for b in range(n_bins) if (bins == b).any()))


def save_calibrator(cal: PlattCalibrator, path: str | Path = CALIBRATOR_FILE) -> None:
    joblib.dump(cal.lr_, path)  # se guarda solo la regresión logística


def load_calibrator(path: str | Path = CALIBRATOR_FILE) -> PlattCalibrator:
    cal = PlattCalibrator()
    cal.lr_ = joblib.load(path)
    return cal


def main() -> None:
    from src import preprocessing
    from src.predict import predict_raw

    df = preprocessing.load_data()
    _, X_val, _, _, y_val, _ = preprocessing.split_data(df)
    cal = PlattCalibrator().fit(predict_raw(X_val), y_val)
    save_calibrator(cal)
    print(f"Calibrador guardado en {CALIBRATOR_FILE}")


if __name__ == "__main__":
    main()

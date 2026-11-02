"""Pruebas de la calibración de probabilidades (src/calibration.py)."""
import numpy as np

from src.calibration import (PlattCalibrator, expected_calibration_error,
                             load_calibrator, save_calibrator)


def _sobreestimadas(n=4000, seed=0):
    """Probabilidades verdaderas y una versión inflada (como con class_weight)."""
    rng = np.random.default_rng(seed)
    p_true = rng.uniform(0.01, 0.6, n)
    y = (rng.uniform(size=n) < p_true).astype(int)
    p_inflada = np.clip(p_true * 1.6, 0, 0.99)
    return p_inflada, y


def test_ece_cero_si_perfecta():
    probs = np.array([0.2] * 5 + [0.8] * 5)
    y = np.array([1, 0, 0, 0, 0, 1, 1, 1, 1, 0])
    assert expected_calibration_error(y, probs) < 1e-9


def test_platt_reduce_error_de_calibracion():
    p, y = _sobreestimadas()
    cal = PlattCalibrator().fit(p, y)
    assert expected_calibration_error(y, cal.transform(p)) < expected_calibration_error(y, p)
    assert abs(cal.transform(p).mean() - y.mean()) < 0.02


def test_guardar_y_cargar(tmp_path):
    p, y = _sobreestimadas()
    cal = PlattCalibrator().fit(p, y)
    path = tmp_path / "cal.joblib"
    save_calibrator(cal, path)
    assert np.allclose(load_calibrator(path).transform(p), cal.transform(p))

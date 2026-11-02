"""Pruebas del modelo (src/model.py) y de la inferencia (src/predict.py)."""
import numpy as np
import pytest

from src import config, preprocessing
from src.model import build_mlp, get_class_weights


def test_arquitectura_salida_probabilidad():
    model = build_mlp(16)
    out = model.predict(np.random.rand(5, 16), verbose=0)
    assert out.shape == (5, 1)
    assert ((out >= 0) & (out <= 1)).all()


def test_pesos_de_clase_favorecen_minoritaria():
    y = np.r_[np.ones(20), np.zeros(80)].astype(int)
    w = get_class_weights(y)
    assert w[1] > w[0]


@pytest.mark.skipif(not config.MODEL_FILE.exists(), reason="Modelo no entrenado")
def test_inferencia_con_modelo_guardado():
    from src.predict import predict_proba

    df = preprocessing.load_data().head(20)
    probs = predict_proba(df)
    assert len(probs) == 20
    assert probs.between(0, 1).all()

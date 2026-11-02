"""Pruebas del preprocesamiento (src/preprocessing.py).

Se usa un dataset sintético pequeño para no depender del archivo real.
"""
import numpy as np
import pandas as pd
import pytest

from src import preprocessing


@pytest.fixture
def df():
    rng = np.random.default_rng(0)
    n = 200
    return pd.DataFrame({
        "RowNumber": range(n),
        "CustomerId": range(1000, 1000 + n),
        "Surname": ["X"] * n,
        "CreditScore": rng.integers(350, 850, n),
        "Geography": rng.choice(["France", "Germany", "Spain"], n),
        "Gender": rng.choice(["Male", "Female"], n),
        "Age": rng.integers(18, 90, n),
        "Tenure": rng.integers(0, 11, n),
        "Balance": rng.uniform(0, 250_000, n),
        "NumOfProducts": rng.integers(1, 5, n),
        "Complain": rng.integers(0, 2, n),
        "Exited": np.r_[np.ones(40), np.zeros(160)].astype(int),
    })


def test_columnas_excluyen_ids_objetivo_y_fuga(df):
    cols = preprocessing.get_feature_columns(df)
    usadas = cols["numeric"] + cols["binary"] + cols["categorical"]
    for c in ["RowNumber", "CustomerId", "Surname", "Exited", "Complain"]:
        assert c not in usadas


def test_split_estratificado(df):
    X_tr, X_va, X_te, y_tr, y_va, y_te = preprocessing.split_data(df)
    assert len(X_tr) + len(X_va) + len(X_te) == len(df)
    for y in (y_tr, y_va, y_te):
        assert y.mean() == pytest.approx(0.20, abs=0.03)


def test_sin_objetivo_lanza_error(df):
    with pytest.raises(ValueError):
        preprocessing.split_data(df.drop(columns=["Exited"]))


def test_preprocesador_se_ajusta_solo_con_train(df):
    X_tr, X_va, *_ = preprocessing.split_data(df)
    prep = preprocessing.fit_preprocessor(X_tr)
    Xt = prep.transform(X_tr)
    names = list(prep.get_feature_names_out())

    # 5 numéricas + 3 países + 2 géneros
    assert Xt.shape[1] == 10
    assert "Complain" not in names
    # Las numéricas quedan con media ~0 en train (escalado ajustado en train)
    num_idx = [names.index(c) for c in ["CreditScore", "Age", "Balance"]]
    assert np.allclose(Xt[:, num_idx].mean(axis=0), 0, atol=1e-9)
    # Validación se transforma sin errores con el mismo preprocesador
    assert prep.transform(X_va).shape[1] == 10


def test_guardar_y_cargar(df, tmp_path):
    X_tr, *_ = preprocessing.split_data(df)
    prep = preprocessing.fit_preprocessor(X_tr)
    path = tmp_path / "prep.joblib"
    preprocessing.save_preprocessor(prep, path)
    loaded = preprocessing.load_preprocessor(path)
    assert np.allclose(loaded.transform(X_tr), prep.transform(X_tr))

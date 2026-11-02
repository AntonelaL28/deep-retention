"""Explicación individual (SHAP) de la probabilidad de fuga de un cliente."""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
import shap

from src import config, preprocessing
from src.predict import load_artifacts


def _original(col: str) -> str:
    for c in config.CATEGORICAL_FEATURES:
        if col.startswith(c + "_"):
            return c
    return col


@lru_cache(maxsize=1)
def _explainer():
    model, prep = load_artifacts()
    df = preprocessing.load_data()
    X_train, *_ = preprocessing.split_data(df)
    bg = prep.transform(X_train.sample(100, random_state=config.RANDOM_STATE))
    names = list(prep.get_feature_names_out())
    f = lambda x: model(x, training=False).numpy().ravel()
    return shap.PermutationExplainer(f, shap.maskers.Independent(bg), feature_names=names,
                                     seed=config.RANDOM_STATE), prep, names


def explain_client(client: pd.DataFrame) -> pd.Series:
    """Contribución SHAP de cada variable (agrupando one-hot) para un cliente."""
    explainer, prep, names = _explainer()
    sv = explainer(prep.transform(client), max_evals=2 * len(names) + 1, silent=True)
    values = pd.Series(np.asarray(sv.values)[0], index=names)
    return values.groupby(_original).sum().sort_values(key=abs, ascending=False)

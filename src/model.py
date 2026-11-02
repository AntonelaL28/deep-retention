"""Arquitectura y entrenamiento del Perceptrón Multicapa (MLP)."""
from __future__ import annotations

import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras

from src import config


def build_mlp(n_features: int,
              hidden_units=config.HIDDEN_UNITS,
              dropout_rates=config.DROPOUT_RATES,
              learning_rate: float = config.LEARNING_RATE) -> keras.Model:
    """MLP: Dense+BatchNorm+ReLU+Dropout -> Dense+ReLU+Dropout -> Sigmoide."""
    inputs = keras.Input(shape=(n_features,), name="variables_cliente")

    x = keras.layers.Dense(hidden_units[0], name="densa_1")(inputs)
    x = keras.layers.BatchNormalization(name="batch_norm")(x)
    x = keras.layers.Activation("relu", name="relu_1")(x)
    x = keras.layers.Dropout(dropout_rates[0], name="dropout_1")(x)

    x = keras.layers.Dense(hidden_units[1], activation="relu", name="densa_2")(x)
    x = keras.layers.Dropout(dropout_rates[1], name="dropout_2")(x)

    outputs = keras.layers.Dense(1, activation="sigmoid", name="prob_fuga")(x)

    model = keras.Model(inputs, outputs, name="deep_retention_mlp")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="roc_auc"),
                 keras.metrics.AUC(curve="PR", name="pr_auc")],
    )
    return model


def get_class_weights(y) -> dict[int, float]:
    """Pesos 'balanced': la clase minoritaria (abandona) pesa más en la pérdida."""
    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    return {int(c): float(w) for c, w in zip(classes, weights)}


def train_mlp(model: keras.Model, X_train, y_train, X_val, y_val,
              batch_size: int = config.BATCH_SIZE,
              max_epochs: int = config.MAX_EPOCHS,
              patience: int = config.EARLY_STOPPING_PATIENCE,
              verbose: int = 2):
    """Entrena con class_weight y Early Stopping sobre val_loss."""
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=patience, restore_best_weights=True)

    return model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=max_epochs,
        batch_size=batch_size,
        class_weight=get_class_weights(y_train),
        callbacks=[early_stopping],
        verbose=verbose,
    )

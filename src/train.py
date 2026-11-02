"""Entrenamiento completo: datos -> preprocesador -> MLP -> archivos guardados.

Uso:  python -m src.train
"""
import json

import matplotlib.pyplot as plt
from tensorflow import keras

from src import calibration, config, preprocessing
from src.model import build_mlp, get_class_weights, train_mlp

HISTORY_FILE = config.MODELS_DIR / "training_history.json"


def plot_history(history: dict, path) -> None:
    epochs = range(1, len(history["loss"]) + 1)
    best = history["val_loss"].index(min(history["val_loss"])) + 1
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, metric, title in [(axes[0], "loss", "Pérdida (binary cross-entropy)"),
                              (axes[1], "roc_auc", "ROC-AUC")]:
        ax.plot(epochs, history[metric], color="#2a78d6", linewidth=2, label="Entrenamiento")
        ax.plot(epochs, history[f"val_{metric}"], color="#eb6834", linewidth=2, label="Validación")
        ax.axvline(best, color="#52514e", linestyle="--", linewidth=1)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Época")
        ax.grid(color="#e6e5e0", linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].plot([], [], color="#52514e", linestyle="--", linewidth=1, label=f"Mejor época ({best})")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    keras.utils.set_random_seed(config.RANDOM_STATE)

    df = preprocessing.load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessing.split_data(df)

    # Preprocesador ajustado solo con train
    prep = preprocessing.fit_preprocessor(X_train)
    Xtr, Xva = prep.transform(X_train), prep.transform(X_val)

    model = build_mlp(Xtr.shape[1])
    model.summary()
    print("Pesos de clase:", get_class_weights(y_train))

    history = train_mlp(model, Xtr, y_train, Xva, y_val).history

    # Guardar artefactos
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(config.MODEL_FILE)
    preprocessing.save_preprocessor(prep)
    HISTORY_FILE.write_text(json.dumps(history, indent=2))
    plot_history(history, config.FIGURES_DIR / "05_curvas_entrenamiento.png")

    # Calibración de probabilidades con validación
    cal = calibration.PlattCalibrator().fit(model.predict(Xva, verbose=0).ravel(), y_val)
    calibration.save_calibrator(cal)

    best = history["val_loss"].index(min(history["val_loss"]))
    print(f"\nÉpocas: {len(history['loss'])} | mejor época: {best + 1}")
    print(f"Validación -> loss {history['val_loss'][best]:.4f} | "
          f"ROC-AUC {history['val_roc_auc'][best]:.4f} | "
          f"PR-AUC {history['val_pr_auc'][best]:.4f}")
    print(f"Modelo guardado en {config.MODEL_FILE}")


if __name__ == "__main__":
    main()

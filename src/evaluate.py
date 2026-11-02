"""Evaluación del modelo sobre el conjunto de prueba.

Genera métricas (reports/metrics.json), curvas ROC/PR, calibración,
matriz de confusión y explicabilidad con SHAP.

Uso:  python -m src.evaluate
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             classification_report, confusion_matrix,
                             precision_recall_curve, roc_auc_score, roc_curve)

from src import calibration, config, preprocessing
from src.predict import load_artifacts, predict_proba, predict_raw

BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#52514e"
METRICS_FILE = config.REPORTS_DIR / "metrics.json"
SHAP_FILE = config.REPORTS_DIR / "shap_importance.csv"
N_SHAP = 500  # clientes de prueba explicados con SHAP


def _style(ax):
    ax.grid(color="#e6e5e0", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def _save(fig, name):
    fig.savefig(config.FIGURES_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_roc_pr(y, probs, roc_auc, pr_auc):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    fpr, tpr, _ = roc_curve(y, probs)
    a1.plot(fpr, tpr, color=BLUE, linewidth=2, label=f"MLP (AUC = {roc_auc:.3f})")
    a1.plot([0, 1], [0, 1], color=GRAY, linestyle="--", linewidth=1, label="Azar")
    a1.set(title="Curva ROC", xlabel="Tasa de falsos positivos", ylabel="Tasa de verdaderos positivos")
    prec, rec, _ = precision_recall_curve(y, probs)
    a2.plot(rec, prec, color=BLUE, linewidth=2, label=f"MLP (AP = {pr_auc:.3f})")
    a2.axhline(np.mean(y), color=GRAY, linestyle="--", linewidth=1, label=f"Azar ({np.mean(y):.3f})")
    a2.set(title="Curva Precision-Recall", xlabel="Recall", ylabel="Precision")
    for ax in (a1, a2):
        _style(ax)
        ax.legend(frameon=False, loc="lower right" if ax is a1 else "upper right")
        ax.title.set_fontweight("bold")
    fig.tight_layout()
    _save(fig, "06_curvas_roc_pr.png")


def plot_calibration(y, raw, cal, n_bins=10):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], color=GRAY, linestyle="--", linewidth=1, label="Calibración perfecta")
    for probs, color, label in [(raw, ORANGE, "Red sin calibrar"), (cal, BLUE, "Red calibrada (Platt)")]:
        bins = np.minimum((probs * n_bins).astype(int), n_bins - 1)
        pts = [(probs[bins == b].mean(), y[bins == b].mean()) for b in range(n_bins) if (bins == b).sum() >= 10]
        ax.plot(*zip(*pts), marker="o", markersize=6, color=color, linewidth=2, label=label)
    ax.set(title="Curva de calibración", xlabel="Probabilidad predicha", ylabel="Tasa real de abandono",
           xlim=(0, 1), ylim=(0, 1))
    ax.title.set_fontweight("bold")
    _style(ax)
    ax.legend(frameon=False)
    _save(fig, "07_calibracion.png")


def plot_confusion(cm):
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(cm, cmap=plt.matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#f0efec", BLUE]))
    labels = ["Se queda", "Abandona"]
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set(xlabel="Predicción", ylabel="Real")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=12,
                    color="white" if cm[i, j] > cm.max() / 2 else "#0b0b0b")
    ax.set_title("Matriz de confusión (red, umbral 0.5)", fontweight="bold")
    _save(fig, "08_matriz_confusion.png")


def shap_analysis(X_train, X_test):
    """SHAP (Permutation) sobre la salida de la red; agrupa las columnas one-hot."""
    model, prep = load_artifacts()
    names = list(prep.get_feature_names_out())
    bg = prep.transform(X_train.sample(100, random_state=config.RANDOM_STATE))
    Xs = prep.transform(X_test.sample(N_SHAP, random_state=config.RANDOM_STATE))

    f = lambda x: model(x, training=False).numpy().ravel()
    explainer = shap.PermutationExplainer(f, shap.maskers.Independent(bg), feature_names=names,
                                       seed=config.RANDOM_STATE)
    sv = explainer(Xs, max_evals=2 * len(names) + 1, silent=True)

    # Beeswarm por columna
    plt.figure()
    shap.plots.beeswarm(sv, max_display=16, show=False)
    plt.title("Impacto de cada variable en la probabilidad de fuga (SHAP)", fontweight="bold")
    plt.xlabel("Valor SHAP (impacto en la probabilidad de fuga)")
    plt.gcf().savefig(config.FIGURES_DIR / "09_shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    # Importancia agregada por variable original
    def original(col):
        for c in config.CATEGORICAL_FEATURES:
            if col.startswith(c + "_"):
                return c
        return col
    imp = (pd.DataFrame(np.abs(sv.values), columns=names).mean()
           .groupby(original).sum().sort_values())
    imp.rename("mean_abs_shap").sort_values(ascending=False).to_csv(SHAP_FILE)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(imp.index, imp.values, color=BLUE, height=0.6)
    for i, v in enumerate(imp.values):
        ax.text(v, i, f" {v:.3f}", va="center", color="#0b0b0b", fontsize=8)
    ax.set(title="Importancia de las variables (media |SHAP|)", xlabel="Impacto medio en la probabilidad")
    ax.title.set_fontweight("bold")
    _style(ax)
    ax.grid(axis="y", visible=False)
    _save(fig, "10_shap_importancia.png")
    return imp.sort_values(ascending=False)


def main() -> None:
    df = preprocessing.load_data()
    X_train, _, X_test, _, _, y_test = preprocessing.split_data(df)
    y = y_test.to_numpy()

    raw = predict_raw(X_test).to_numpy()
    cal = predict_proba(X_test).to_numpy()

    roc_auc = roc_auc_score(y, cal)
    pr_auc = average_precision_score(y, cal)
    y_pred = (raw >= 0.5).astype(int)
    cm = confusion_matrix(y, y_pred)
    report = classification_report(y, y_pred, output_dict=True)

    plot_roc_pr(y, cal, roc_auc, pr_auc)
    plot_calibration(y, raw, cal)
    plot_confusion(cm)
    imp = shap_analysis(X_train, X_test)

    metrics = {
        "conjunto": "prueba",
        "n_clientes": int(len(y)),
        "tasa_real_abandono": float(y.mean()),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "brier_sin_calibrar": float(brier_score_loss(y, raw)),
        "brier_calibrado": float(brier_score_loss(y, cal)),
        "ece_sin_calibrar": calibration.expected_calibration_error(y, raw),
        "ece_calibrado": calibration.expected_calibration_error(y, cal),
        "prob_media_sin_calibrar": float(raw.mean()),
        "prob_media_calibrada": float(cal.mean()),
        "umbral_0.5": {
            "matriz_confusion": cm.tolist(),
            "recall_abandono": report["1"]["recall"],
            "precision_abandono": report["1"]["precision"],
            "f1_abandono": report["1"]["f1-score"],
            "accuracy": report["accuracy"],
        },
        "importancia_shap": imp.round(4).to_dict(),
    }
    METRICS_FILE.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))

    print(f"ROC-AUC {roc_auc:.4f} | PR-AUC {pr_auc:.4f}")
    print(f"Brier {metrics['brier_sin_calibrar']:.4f} -> {metrics['brier_calibrado']:.4f} | "
          f"ECE {metrics['ece_sin_calibrar']:.4f} -> {metrics['ece_calibrado']:.4f}")
    print("Importancia SHAP:\n", imp.round(4).to_string())
    print(f"Métricas guardadas en {METRICS_FILE}")


if __name__ == "__main__":
    main()

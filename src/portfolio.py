"""Aplicación del sistema completo a la cartera: red neuronal + regla financiera.

Uso:  python -m src.portfolio
"""
from __future__ import annotations

import json

import pandas as pd

from src import config, finance, preprocessing
from src.predict import predict_proba

SCORED_FILE = config.PROCESSED_DATA_DIR / "cartera_evaluada.csv"
SUMMARY_FILE = config.REPORTS_DIR / "resumen_cartera.json"
MATRIX_FILE = config.REPORTS_DIR / "matriz_decision.csv"


def score_portfolio(df: pd.DataFrame,
                    catalog: dict | None = None,
                    margin_params: dict | None = None,
                    tier_thresholds: tuple[float, float] | None = None) -> pd.DataFrame:
    """Probabilidad calibrada, margen y decisión comercial para cada cliente."""
    out = df.copy()
    out["prob_churn"] = predict_proba(out)
    out["margin"] = finance.estimate_margin(out, margin_params or config.MARGIN_PARAMS)
    return finance.decide_retention(out, catalog=catalog, tier_thresholds=tier_thresholds)


def main() -> None:
    df = preprocessing.load_data().drop(columns=[config.TARGET, "Complain"])
    thresholds = finance.compute_tier_thresholds(
        finance.estimate_margin(df, config.MARGIN_PARAMS))
    scored = score_portfolio(df, tier_thresholds=thresholds)

    summary = finance.portfolio_summary(scored, balance_col="Balance")
    summary["umbrales_valor"] = {"superior": thresholds[0], "medio": thresholds[1]}
    summary["sin_filtro_rentabilidad"] = finance.without_profit_filter(scored)
    matrix = finance.segment_matrix(scored)

    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    scored.to_csv(SCORED_FILE, index=False)
    SUMMARY_FILE.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    matrix.to_csv(MATRIX_FILE, index=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False, default=float))
    print(matrix.to_string(index=False))


if __name__ == "__main__":
    main()

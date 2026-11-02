"""Lógica financiera del sistema Deep Retention.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config

SEGMENTS = {
    "A": ("A: Alto Valor / Alto Riesgo", "Retención Prioritaria"),
    "B": ("B: Medio / Riesgo Moderado", "Retención Automatizada"),
    "C": ("C: No Rentable", "Salida Permitida"),
    "D": ("D: Estable", "Sin Acción Requerida"),
}
NO_INCENTIVE = "Sin intervención"

# Formula base
def expected_net_gain(prob_churn, effectiveness, margin, cost):
    """Ganancia Neta Esperada = P(fuga) × Efectividad × Margen − Costo.

    Acepta escalares o arreglos (se opera elemento a elemento).
    """
    return (np.asarray(prob_churn) * effectiveness * np.asarray(margin)) - cost


# Clasificaciones auxiliares
def risk_level(prob_churn,
               high=config.HIGH_RISK_THRESHOLD,
               low=config.LOW_RISK_THRESHOLD) -> np.ndarray:
    """Nivel de riesgo: 'Alto' (>= high), 'Moderado' o 'Bajo' (< low)."""
    p = np.asarray(prob_churn, dtype=float)
    return np.select([p >= high, p >= low], ["Alto", "Moderado"], default="Bajo")


def compute_tier_thresholds(balances,
                            q_high=config.TIER_HIGH_QUANTILE,
                            q_low=config.TIER_LOW_QUANTILE) -> tuple[float, float]:
    """Calcula los umbrales de saldo (alto, bajo) a partir de percentiles."""
    b = pd.Series(balances, dtype=float)
    return float(b.quantile(q_high)), float(b.quantile(q_low))


def value_tier(balance, margin, high_threshold: float, low_threshold: float) -> np.ndarray:
    """Nivel de valor patrimonial del cliente.

    'Deficitario' si su margen es <= 0 (le cuesta dinero al banco);
    si no, 'Superior', 'Medio' o 'Bajo' según su saldo.
    """
    b = np.asarray(balance, dtype=float)
    m = np.asarray(margin, dtype=float)
    return np.select(
        [m <= 0, b >= high_threshold, b >= low_threshold],
        ["Deficitario", "Superior", "Medio"],
        default="Bajo",
    )

# Motor de decisión
def decide_retention(df: pd.DataFrame,
                     prob_col: str = "prob_churn",
                     balance_col: str = "balance",
                     margin_col: str = "margin",
                     catalog: dict | None = None,
                     high_risk: float = config.HIGH_RISK_THRESHOLD,
                     low_risk: float = config.LOW_RISK_THRESHOLD,
                     tier_thresholds: tuple[float, float] | None = None) -> pd.DataFrame:
    catalog = catalog or config.INCENTIVE_CATALOG
    out = df.copy()

    p = out[prob_col].to_numpy(dtype=float)
    balance = out[balance_col].to_numpy(dtype=float)
    margin = out[margin_col].to_numpy(dtype=float)

    if tier_thresholds is None:
        tier_thresholds = compute_tier_thresholds(balance)
    high_t, low_t = tier_thresholds

    out["nivel_riesgo"] = risk_level(p, high_risk, low_risk)
    out["nivel_valor"] = value_tier(balance, margin, high_t, low_t)
    tiers = out["nivel_valor"].to_numpy()

    # Ganancia de cada incentivo (−inf si no está permitido para el nivel)
    codes = list(catalog.keys())
    gains = np.full((len(out), len(codes)), -np.inf)
    for j, code in enumerate(codes):
        inc = catalog[code]
        allowed = np.isin(tiers, inc["tiers"])
        gain = expected_net_gain(p, inc["effectiveness"], margin, inc["cost"])
        gains[:, j] = np.where(allowed, gain, -np.inf)

    best_idx = gains.argmax(axis=1)
    best_gain = gains[np.arange(len(out)), best_idx]

    stable = p < low_risk
    act = (~stable) & (best_gain > 0)

    # Segmento
    seg = np.where(stable, "D",
          np.where(~act, "C",
          np.where(tiers == "Superior", "A", "B")))

    best_codes = np.array(codes)[best_idx]
    out["incentivo"] = np.where(act, [catalog[c]["name"] for c in best_codes], NO_INCENTIVE)
    out["costo_incentivo"] = np.where(act, [catalog[c]["cost"] for c in best_codes], 0.0)
    out["efectividad"] = np.where(act, [catalog[c]["effectiveness"] for c in best_codes], 0.0)
    out["ganancia_neta_esperada"] = np.where(act, best_gain, 0.0)
    out["segmento"] = [SEGMENTS[s][0] for s in seg]
    out["decision"] = [SEGMENTS[s][1] for s in seg]
    return out


# Resumen de cartera (dashboard e informe ejecutivo)
def portfolio_summary(decided: pd.DataFrame,
                      prob_col: str = "prob_churn",
                      balance_col: str = "balance",
                      margin_col: str = "margin") -> dict:
    """KPIs de la cartera a partir de la salida de `decide_retention`."""
    intervened = decided["decision"].isin(
        [SEGMENTS["A"][1], SEGMENTS["B"][1]])
    high_risk = decided["nivel_riesgo"] == "Alto"

    investment = float(decided["costo_incentivo"].sum())
    net_gain = float(decided["ganancia_neta_esperada"].sum())

    # Margen que se espera conservar gracias a las intervenciones
    expected_rescued_margin = float(
        (decided.loc[intervened, prob_col]
         * decided.loc[intervened, "efectividad"]
         * decided.loc[intervened, margin_col]).sum())
    # Clientes que se espera retener (suma de probabilidades)
    expected_retained = float(
        (decided.loc[intervened, prob_col]
         * decided.loc[intervened, "efectividad"]).sum())

    return {
        "clientes_analizados": int(len(decided)),
        "clientes_alto_riesgo": int(high_risk.sum()),
        "clientes_a_intervenir": int(intervened.sum()),
        "salidas_permitidas": int((decided["decision"] == SEGMENTS["C"][1]).sum()),
        "capital_en_riesgo": float(decided.loc[high_risk, balance_col].sum()),
        "capital_rescatable": float(decided.loc[intervened, balance_col].sum()),
        "inversion_total": investment,
        "margen_rescatado_esperado": expected_rescued_margin,
        "ganancia_neta_total": net_gain,
        "clientes_retenidos_esperados": expected_retained,
        "roi": (net_gain / investment) if investment > 0 else 0.0,
    }


def segment_matrix(decided: pd.DataFrame) -> pd.DataFrame:
    """Matriz de decisión comercial agregada por segmento (para el informe)."""
    return (decided.groupby(["segmento", "decision"], as_index=False)
            .agg(clientes=("decision", "size"),
                 costo_total=("costo_incentivo", "sum"),
                 ganancia_neta=("ganancia_neta_esperada", "sum"))
            .sort_values("segmento")
            .reset_index(drop=True))

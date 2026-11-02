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


def estimate_margin(df: pd.DataFrame, params: dict = config.MARGIN_PARAMS) -> pd.Series:
    """Margen anual neto que el cliente genera al banco (supuestos en config)."""
    tx_rate = np.where(df["IsActiveMember"] == 1,
                       params["transaction_rate_active"], params["transaction_rate_inactive"])
    card_income = df["HasCrCard"] * df["Card Type"].map(params["card_income"]).fillna(0)
    margin = (df["Balance"] * params["balance_spread"]
              + df["NumOfProducts"] * params["fee_per_product"]
              + card_income
              + df["EstimatedSalary"] * tx_rate
              - params["operating_cost"])
    return margin.rename("margin")


def compute_tier_thresholds(margins,
                            q_high=config.TIER_HIGH_QUANTILE,
                            q_low=config.TIER_LOW_QUANTILE) -> tuple[float, float]:
    """Umbrales de margen (alto, bajo) según percentiles de los clientes rentables."""
    m = pd.Series(margins, dtype=float)
    m = m[m > 0]
    return float(m.quantile(q_high)), float(m.quantile(q_low))


def value_tier(margin, high_threshold: float, low_threshold: float) -> np.ndarray:
    """'Deficitario' si margen <= 0; si no 'Superior', 'Medio' o 'Bajo'."""
    m = np.asarray(margin, dtype=float)
    return np.select(
        [m <= 0, m >= high_threshold, m >= low_threshold],
        ["Deficitario", "Superior", "Medio"],
        default="Bajo",
    )

# Motor de decisión
def decide_retention(df: pd.DataFrame,
                     prob_col: str = "prob_churn",
                     margin_col: str = "margin",
                     catalog: dict | None = None,
                     high_risk: float = config.HIGH_RISK_THRESHOLD,
                     low_risk: float = config.LOW_RISK_THRESHOLD,
                     tier_thresholds: tuple[float, float] | None = None) -> pd.DataFrame:
    """Aplica la regla de ganancia neta: asigna incentivo, segmento y decisión."""
    catalog = catalog or config.INCENTIVE_CATALOG
    out = df.copy()

    p = out[prob_col].to_numpy(dtype=float)
    margin = out[margin_col].to_numpy(dtype=float)

    if tier_thresholds is None:
        tier_thresholds = compute_tier_thresholds(margin)
    high_t, low_t = tier_thresholds

    out["nivel_riesgo"] = risk_level(p, high_risk, low_risk)
    out["nivel_valor"] = value_tier(margin, high_t, low_t)
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
    # Saldo que se espera conservar gracias a las intervenciones
    expected_rescued_capital = float(
        (decided.loc[intervened, prob_col]
         * decided.loc[intervened, "efectividad"]
         * decided.loc[intervened, balance_col]).sum())
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
        "capital_clientes_intervenidos": float(decided.loc[intervened, balance_col].sum()),
        "capital_rescatable": expected_rescued_capital,
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


def without_profit_filter(decided: pd.DataFrame,
                          fallback_code: str = "BONIFICACION",
                          prob_col: str = "prob_churn",
                          margin_col: str = "margin",
                          catalog: dict | None = None) -> dict:
    """Escenario de comparación: se retiene a todo cliente en riesgo, rentable o no.

    Los clientes 'Salida Permitida' reciben el incentivo más barato (fallback_code).
    """
    inc = (catalog or config.INCENTIVE_CATALOG)[fallback_code]
    exits = decided["decision"] == SEGMENTS["C"][1]
    extra_gain = expected_net_gain(decided.loc[exits, prob_col], inc["effectiveness"],
                                   decided.loc[exits, margin_col], inc["cost"])
    investment = float(decided["costo_incentivo"].sum() + inc["cost"] * exits.sum())
    net_gain = float(decided["ganancia_neta_esperada"].sum() + extra_gain.sum())
    return {
        "clientes_intervenidos": int((decided["costo_incentivo"] > 0).sum() + exits.sum()),
        "inversion_total": investment,
        "ganancia_neta_total": net_gain,
        "roi": net_gain / investment if investment > 0 else 0.0,
        "perdida_evitada_por_filtro": float(-extra_gain.sum()),
        "inversion_evitada_por_filtro": float(inc["cost"] * exits.sum()),
    }

"""Pruebas unitarias de la lógica financiera (src/finance.py)."""
import pandas as pd
import pytest

from src import finance

CATALOG = {
    "VIP": {"name": "Gestor VIP", "cost": 100.0, "effectiveness": 0.5,
            "tiers": ["Superior"]},
    "BONO": {"name": "Bono", "cost": 20.0, "effectiveness": 0.2,
             "tiers": ["Superior", "Medio", "Bajo"]},
}
THRESHOLDS = (1_500.0, 500.0)  # margen alto, margen bajo


def _decide(rows):
    df = pd.DataFrame(rows, columns=["prob_churn", "balance", "margin"])
    return finance.decide_retention(df, catalog=CATALOG,
                                    tier_thresholds=THRESHOLDS)


def test_formula_ganancia_neta():
    # 0.8 * 0.4 * 500 - 50 = 110
    assert finance.expected_net_gain(0.8, 0.4, 500, 50) == pytest.approx(110)


def test_cliente_rentable_alto_valor_es_prioritario():
    r = _decide([[0.9, 50_000, 2_000]]).iloc[0]
    # VIP: 0.9*0.5*2000-100 = 800 ; Bono: 0.9*0.2*2000-20 = 340 -> VIP
    assert r["decision"] == "Retención Prioritaria"
    assert r["incentivo"] == "Gestor VIP"
    assert r["ganancia_neta_esperada"] == pytest.approx(800)


def test_cliente_deficitario_es_salida_permitida():
    r = _decide([[0.9, 50_000, -300]]).iloc[0]
    assert r["nivel_valor"] == "Deficitario"
    assert r["decision"] == "Salida Permitida"
    assert r["costo_incentivo"] == 0


def test_ganancia_negativa_bloquea_gasto():
    # Bajo: solo Bono -> 0.7*0.2*100-20 = -6 < 0
    r = _decide([[0.7, 5_000, 100]]).iloc[0]
    assert r["decision"] == "Salida Permitida"
    assert r["ganancia_neta_esperada"] == 0


def test_cliente_estable_sin_accion():
    r = _decide([[0.1, 50_000, 5_000]]).iloc[0]
    assert r["decision"] == "Sin Acción Requerida"
    assert r["costo_incentivo"] == 0


def test_rango_intermedio_decide_la_formula():
    # p = 0.5 (entre 0.30 y 0.65), Medio: 0.5*0.2*1000-20 = 80 > 0
    r = _decide([[0.5, 5_000, 1_000]]).iloc[0]
    assert r["nivel_riesgo"] == "Moderado"
    assert r["decision"] == "Retención Automatizada"


def test_resumen_de_cartera_y_roi():
    decided = _decide([
        [0.9, 50_000, 2_000],   # A: ganancia 800, costo 100
        [0.5, 5_000, 1_000],    # B: ganancia 80,  costo 20
        [0.9, 50_000, -300],    # C
        [0.1, 50_000, 5_000],   # D
    ])
    s = finance.portfolio_summary(decided)
    assert s["clientes_analizados"] == 4
    assert s["clientes_a_intervenir"] == 2
    assert s["inversion_total"] == pytest.approx(120)
    assert s["ganancia_neta_total"] == pytest.approx(880)
    assert s["roi"] == pytest.approx(880 / 120)
    assert s["capital_en_riesgo"] == pytest.approx(100_000)


def test_estimar_margen():
    import pandas as pd
    df = pd.DataFrame({"Balance": [100_000, 0], "NumOfProducts": [2, 1],
                       "HasCrCard": [1, 0], "Card Type": ["GOLD", "SILVER"],
                       "EstimatedSalary": [50_000, 10_000], "IsActiveMember": [1, 0]})
    m = finance.estimate_margin(df)
    # 2000 + 100 + 80 + 200 - 180 = 2200 ; 0 + 50 + 0 + 10 - 180 = -120
    assert m.tolist() == pytest.approx([2200, -120])


def test_filtro_evita_perdidas():
    decided = _decide([[0.9, 50_000, 2_000], [0.9, 0, -300], [0.1, 0, 100]])
    r = finance.without_profit_filter(decided, fallback_code="BONO", catalog=CATALOG)
    # El deficitario recibiría Bono: 0.9*0.2*(-300) - 20 = -74
    assert r["inversion_evitada_por_filtro"] == pytest.approx(20)
    assert r["perdida_evitada_por_filtro"] == pytest.approx(74)
    assert r["ganancia_neta_total"] == pytest.approx(800 - 74)

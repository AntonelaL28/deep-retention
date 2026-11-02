"""Aplicación Deep Retention.

Uso:  streamlit run app/app.py
"""
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import config, finance
from src.portfolio import SCORED_FILE, SUMMARY_FILE

st.set_page_config(page_title="Deep Retention", page_icon="🏦", layout="wide")

BLUE, ORANGE, AQUA, YELLOW, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a8984"
DECISION_COLORS = {
    "Retención Prioritaria": BLUE,
    "Retención Automatizada": AQUA,
    "Salida Permitida": ORANGE,
    "Sin Acción Requerida": GRAY,
}
DECISION_ORDER = list(DECISION_COLORS)
CLIENT_COLS = ["CustomerId", "Surname", "CreditScore", "Geography", "Gender", "Age", "Tenure",
               "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary",
               "Card Type"]


# Utilidades
def money(x: float) -> str:
    return f"-${abs(x):,.0f}" if x < 0 else f"${x:,.0f}"


def md_money(x: float) -> str:
    """money() escapado para markdown (evita que $ se lea como fórmula)."""
    return money(x).replace("$", "\\$")


MATRIX_COLS = {"segmento": "Segmento", "decision": "Decisión", "clientes": "Clientes",
               "costo_total": "Costo total", "ganancia_neta": "Ganancia neta"}
MATRIX_FMT = {"Costo total": "${:,.0f}", "Ganancia neta": "${:,.0f}", "Clientes": "{:,}"}


def style_fig(fig, height=320):
    fig.update_layout(template="plotly_white", height=height, margin=dict(l=10, r=10, t=40, b=10),
                      font=dict(size=12), legend_title_text="", title_font_size=14)
    return fig


@st.cache_data
def load_portfolio():
    scored = pd.read_csv(SCORED_FILE)
    summary = json.loads(SUMMARY_FILE.read_text(encoding="utf-8"))
    thresholds = (summary["umbrales_valor"]["superior"], summary["umbrales_valor"]["medio"])
    return scored, thresholds


def apply_rules(scored, catalog=None, high=config.HIGH_RISK_THRESHOLD,
                low=config.LOW_RISK_THRESHOLD, margin_params=None, thresholds=None):
    """Recalcula margen y decisiones con los parámetros indicados."""
    df = scored[CLIENT_COLS + ["prob_churn"]].copy()
    df["margin"] = finance.estimate_margin(df, margin_params or config.MARGIN_PARAMS)
    if thresholds is None:
        thresholds = finance.compute_tier_thresholds(df["margin"])
    return finance.decide_retention(df, catalog=catalog, high_risk=high, low_risk=low,
                                    tier_thresholds=thresholds)


def decision_badge(decision: str) -> str:
    color = DECISION_COLORS[decision]
    return (f"<span style='background:{color};color:white;padding:4px 12px;"
            f"border-radius:12px;font-weight:600'>{decision}</span>")


# Panel de cartera
def page_dashboard(scored):
    st.title("Panel de Cartera")
    st.caption("Riesgo de fuga, inversión y retorno de la estrategia de retención sobre toda la cartera.")

    c1, c2, c3 = st.columns(3)
    geo = c1.multiselect("País", sorted(scored["Geography"].unique()), placeholder="Todos")
    valor = c2.multiselect("Nivel de valor", ["Superior", "Medio", "Bajo", "Deficitario"],
                           placeholder="Todos")
    dec = c3.multiselect("Decisión", DECISION_ORDER, placeholder="Todas")
    df = scored
    if geo:
        df = df[df["Geography"].isin(geo)]
    if valor:
        df = df[df["nivel_valor"].isin(valor)]
    if dec:
        df = df[df["decision"].isin(dec)]
    if df.empty:
        st.warning("No hay clientes con esos filtros.")
        return

    s = finance.portfolio_summary(df, balance_col="Balance")
    at_risk = int((df["nivel_riesgo"] != "Bajo").sum())
    k = st.columns(4)
    k[0].metric("Clientes analizados", f"{s['clientes_analizados']:,}")
    k[1].metric("Clientes en riesgo (P ≥ 0.30)", f"{at_risk:,}")
    k[2].metric("Clientes a intervenir", f"{s['clientes_a_intervenir']:,}")
    k[3].metric("Salidas permitidas", f"{s['salidas_permitidas']:,}")
    k = st.columns(4)
    k[0].metric("Capital en riesgo crítico", money(s["capital_en_riesgo"]),
                help="Saldo de clientes con probabilidad de fuga ≥ 0.65")
    k[1].metric("Capital rescatable esperado", money(s["capital_rescatable"]),
                help="Σ P(fuga) × efectividad × saldo de los clientes intervenidos")
    k[2].metric("Inversión en retención", money(s["inversion_total"]))
    k[3].metric("Ganancia neta esperada", money(s["ganancia_neta_total"]),
                delta=f"ROI {s['roi']:.0%}")
    if s["clientes_a_intervenir"]:
        st.caption(f"Efectividad proyectada: se espera retener **{s['clientes_retenidos_esperados']:,.0f}** "
                   f"de {s['clientes_a_intervenir']:,} clientes intervenidos.")

    st.divider()
    g1, g2 = st.columns(2)
    counts = df["decision"].value_counts().reindex(DECISION_ORDER).fillna(0).reset_index()
    counts.columns = ["decision", "clientes"]
    fig = px.bar(counts, x="clientes", y="decision", orientation="h", color="decision",
                 color_discrete_map=DECISION_COLORS, text="clientes",
                 title="Clientes por decisión del sistema")
    fig.update_layout(showlegend=False, yaxis_title="", xaxis_title="Clientes")
    g1.plotly_chart(style_fig(fig), width="stretch")

    fig = px.histogram(df, x="prob_churn", nbins=40, color_discrete_sequence=[BLUE],
                       title="Distribución de la probabilidad de fuga")
    for th, label in [(config.LOW_RISK_THRESHOLD, "0.30"), (config.HIGH_RISK_THRESHOLD, "0.65")]:
        fig.add_vline(x=th, line_dash="dash", line_color=GRAY, annotation_text=label)
    fig.update_layout(xaxis_title="Probabilidad de fuga (calibrada)", yaxis_title="Clientes",
                      bargap=0.05)
    g2.plotly_chart(style_fig(fig), width="stretch")

    g1, g2 = st.columns(2)
    sample = df.sample(min(len(df), 3000), random_state=0)
    fig = px.scatter(sample, x="prob_churn", y="margin", color="decision",
                     color_discrete_map=DECISION_COLORS, opacity=0.6,
                     category_orders={"decision": DECISION_ORDER},
                     hover_data={"Surname": True, "Balance": ":,.0f", "incentivo": True},
                     title="Riesgo vs. valor del cliente")
    fig.update_traces(marker=dict(size=5))
    fig.update_layout(xaxis_title="Probabilidad de fuga", yaxis_title="Margen anual ($)")
    g1.plotly_chart(style_fig(fig, 360), width="stretch")

    by_geo = df.groupby("Geography").apply(
        lambda g: g.loc[g["nivel_riesgo"] == "Alto", "Balance"].sum(), include_groups=False)
    fig = px.bar(by_geo.reset_index(name="capital"), x="Geography", y="capital", text_auto=".3s",
                 color_discrete_sequence=[ORANGE], title="Capital en riesgo crítico por país")
    fig.update_layout(xaxis_title="", yaxis_title="Saldo ($)")
    g2.plotly_chart(style_fig(fig, 360), width="stretch")

    st.subheader("Matriz de decisión comercial")
    m = finance.segment_matrix(df).rename(columns=MATRIX_COLS)
    st.dataframe(m.style.format(MATRIX_FMT), hide_index=True, width="stretch")

    st.subheader("Clientes prioritarios a contactar")
    top = (df[df["costo_incentivo"] > 0].sort_values("ganancia_neta_esperada", ascending=False)
           [["CustomerId", "Surname", "Geography", "Age", "prob_churn", "margin",
             "incentivo", "costo_incentivo", "ganancia_neta_esperada", "decision"]]
           .rename(columns={"CustomerId": "ID", "Surname": "Apellido", "Geography": "País",
                            "Age": "Edad", "prob_churn": "P(fuga)", "margin": "Margen anual",
                            "incentivo": "Incentivo", "costo_incentivo": "Costo",
                            "ganancia_neta_esperada": "Ganancia neta", "decision": "Decisión"}))
    st.dataframe(top.head(50).style.format({"P(fuga)": "{:.1%}", "Margen anual": "${:,.0f}",
                                            "Costo": "${:,.0f}", "Ganancia neta": "${:,.0f}"}),
                 hide_index=True, width="stretch")
    st.download_button("Descargar lista completa (CSV)", top.to_csv(index=False).encode("utf-8"),
                       "clientes_a_intervenir.csv", "text/csv")


# Buscador por cliente
def show_client_result(row: pd.Series, client: pd.DataFrame, thresholds):
    p, m = row["prob_churn"], row["margin"]
    k = st.columns(4)
    k[0].metric("Probabilidad de fuga", f"{p:.1%}", help="Salida calibrada de la red neuronal")
    k[1].metric("Nivel de riesgo", row["nivel_riesgo"])
    k[2].metric("Margen anual del cliente", money(m))
    k[3].metric("Nivel de valor", row["nivel_valor"])

    st.markdown(f"**Decisión sugerida:** {decision_badge(row['decision'])}", unsafe_allow_html=True)
    st.write("")
    if row["costo_incentivo"] > 0:
        st.markdown(f"**Incentivo asignado:** {row['incentivo']}")
        k = st.columns(3)
        k[0].metric("Costo del incentivo", money(row["costo_incentivo"]))
        k[1].metric("Efectividad esperada", f"{row['efectividad']:.0%}")
        k[2].metric("Ganancia neta esperada", money(row["ganancia_neta_esperada"]))
        st.info(f"Ganancia neta esperada = {p:.3f} × {row['efectividad']:.2f} × {md_money(m)} − "
                f"{md_money(row['costo_incentivo'])} = **{md_money(row['ganancia_neta_esperada'])}** "
                f"> 0 → se autoriza la retención.")
    elif row["decision"] == "Salida Permitida":
        st.warning("Ningún incentivo genera ganancia neta positiva para este cliente: "
                   "retenerlo costaría más de lo que aporta. No se autoriza gasto de fidelización.")
    else:
        st.success(f"Riesgo bajo (< {config.LOW_RISK_THRESHOLD:.2f}): mantenimiento ordinario, sin gasto.")

    rows = []
    for inc in config.INCENTIVE_CATALOG.values():
        allowed = row["nivel_valor"] in inc["tiers"]
        gain = finance.expected_net_gain(p, inc["effectiveness"], m, inc["cost"])
        rows.append({"Incentivo": inc["name"], "Costo": inc["cost"],
                     "Efectividad": inc["effectiveness"],
                     "Permitido para su nivel": "Sí" if allowed else "No",
                     "Ganancia neta esperada": float(gain)})
    with st.expander("Evaluación de todos los incentivos"):
        st.dataframe(pd.DataFrame(rows).style.format({"Costo": "${:,.0f}", "Efectividad": "{:.0%}",
                                                      "Ganancia neta esperada": "${:,.0f}"}),
                     hide_index=True, width="stretch")

    with st.expander("¿Por qué este riesgo? (factores SHAP)", expanded=True):
        try:
            from src.explain import explain_client
            contrib = explain_client(client).head(8).iloc[::-1]
            fig = go.Figure(go.Bar(x=contrib.values, y=contrib.index, orientation="h",
                                   marker_color=[ORANGE if v > 0 else BLUE for v in contrib.values]))
            fig.update_layout(title="Contribución de cada variable a la probabilidad de fuga",
                              xaxis_title="← reduce el riesgo  |  aumenta el riesgo →")
            st.plotly_chart(style_fig(fig, 320), width="stretch")
        except Exception as e:  # la explicación es complementaria
            st.caption(f"No se pudo calcular la explicación: {e}")


def page_search(scored, thresholds):
    st.title("Buscador de Clientes")
    st.caption("Puntaje de la red neuronal, impacto monetario y decisión comercial por cliente.")
    mode = st.radio("Modo", ["Cliente de la cartera", "Nuevo cliente"], horizontal=True)

    if mode == "Cliente de la cartera":
        options = scored["CustomerId"].astype(str) + " — " + scored["Surname"]
        choice = st.selectbox("Buscar por ID o apellido", options, index=None,
                              placeholder="Escriba el ID o el apellido del cliente…")
        if choice is None:
            return
        row = scored.loc[options == choice].iloc[0]
        st.subheader(f"{row['Surname']} · ID {row['CustomerId']}")
        perfil = row[CLIENT_COLS[2:]].to_frame("Valor").T
        st.dataframe(perfil, hide_index=True, width="stretch")
        show_client_result(row, row[CLIENT_COLS].to_frame().T.infer_objects(), thresholds)
        return

    with st.form("nuevo_cliente"):
        c = st.columns(3)
        data = {
            "CustomerId": 0, "Surname": "Nuevo cliente",
            "CreditScore": c[0].number_input("Puntaje de crédito", 300, 850, 650),
            "Geography": c[1].selectbox("País", ["France", "Germany", "Spain"]),
            "Gender": c[2].selectbox("Género", ["Female", "Male"]),
            "Age": c[0].number_input("Edad", 18, 100, 40),
            "Tenure": c[1].number_input("Antigüedad (años)", 0, 10, 5),
            "Balance": c[2].number_input("Saldo ($)", 0.0, 1_000_000.0, 100_000.0, step=1000.0),
            "NumOfProducts": c[0].selectbox("Número de productos", [1, 2, 3, 4]),
            "EstimatedSalary": c[1].number_input("Salario estimado ($)", 0.0, 1_000_000.0,
                                                 80_000.0, step=1000.0),
            "Card Type": c[2].selectbox("Tipo de tarjeta", ["SILVER", "GOLD", "PLATINUM", "DIAMOND"]),
            "HasCrCard": int(c[0].checkbox("Tiene tarjeta de crédito", True)),
            "IsActiveMember": int(c[1].checkbox("Cliente activo", True)),
        }
        submitted = st.form_submit_button("Evaluar cliente", type="primary")
    if not submitted:
        return
    from src.predict import predict_proba
    client = pd.DataFrame([data])
    client["prob_churn"] = predict_proba(client).to_numpy()
    client["margin"] = finance.estimate_margin(client)
    row = finance.decide_retention(client, tier_thresholds=thresholds).iloc[0]
    show_client_result(row, client[CLIENT_COLS], thresholds)


# Simulador de parámetros
def page_simulator(scored, thresholds):
    st.title("Simulador de Parámetros")
    st.caption("Ajuste costos y efectividad de las campañas y compare el impacto en la rentabilidad.")
    if st.button("Restablecer valores"):
        for key in list(st.session_state):
            if key.startswith("sim_"):
                del st.session_state[key]

    catalog = copy.deepcopy(config.INCENTIVE_CATALOG)
    cols = st.columns(len(catalog))
    for col, (code, inc) in zip(cols, catalog.items()):
        with col:
            st.markdown(f"**{inc['name']}**")
            inc["cost"] = st.number_input("Costo ($)", 0.0, 5000.0, inc["cost"], step=10.0,
                                          key=f"sim_cost_{code}")
            inc["effectiveness"] = st.slider("Efectividad", 0.0, 1.0, inc["effectiveness"], 0.05,
                                             key=f"sim_eff_{code}")

    with st.expander("Umbrales de riesgo y supuestos de margen"):
        c = st.columns(2)
        low, high = c[0].slider("Umbrales de riesgo (bajo, alto)", 0.0, 1.0,
                                (config.LOW_RISK_THRESHOLD, config.HIGH_RISK_THRESHOLD), 0.05,
                                key="sim_thresholds")
        mp = copy.deepcopy(config.MARGIN_PARAMS)
        mp["balance_spread"] = c[1].slider("Margen de interés sobre el saldo", 0.0, 0.05,
                                           mp["balance_spread"], 0.0025, format="%.4f",
                                           key="sim_spread")
        mp["operating_cost"] = c[1].number_input("Costo anual de servicio ($)", 0.0, 2000.0,
                                                 mp["operating_cost"], step=10.0, key="sim_opcost")

    margin_changed = mp != config.MARGIN_PARAMS
    sim = apply_rules(scored, catalog, high, low, mp, None if margin_changed else thresholds)
    base_s = finance.portfolio_summary(scored, balance_col="Balance")
    sim_s = finance.portfolio_summary(sim, balance_col="Balance")

    st.subheader("Resultado de la simulación")
    k = st.columns(4)
    for col, key, label, fmt in [
        (k[0], "clientes_a_intervenir", "Clientes a intervenir", lambda v: f"{v:,}"),
        (k[1], "inversion_total", "Inversión", money),
        (k[2], "ganancia_neta_total", "Ganancia neta esperada", money),
        (k[3], "roi", "ROI", lambda v: f"{v:.0%}"),
    ]:
        diff = sim_s[key] - base_s[key]
        delta = (f"{diff:+.0%}" if key == "roi" else f"{diff:+,.0f}") if abs(diff) > 1e-9 else None
        col.metric(label, fmt(sim_s[key]), delta=delta,
                   delta_color="inverse" if key == "inversion_total" else "normal")
    st.caption("Las variaciones se comparan con los parámetros base del sistema.")

    comp = pd.DataFrame({
        "Base": scored["decision"].value_counts(),
        "Simulación": sim["decision"].value_counts(),
    }).reindex(DECISION_ORDER).fillna(0).astype(int)
    c1, c2 = st.columns(2)
    fig = go.Figure([go.Bar(name=n, y=comp.index, x=comp[n], orientation="h", marker_color=color)
                     for n, color in [("Base", GRAY), ("Simulación", BLUE)]])
    fig.update_layout(barmode="group", title="Clientes por decisión", xaxis_title="Clientes",
                      yaxis_autorange="reversed")
    c1.plotly_chart(style_fig(fig), width="stretch")

    code = c2.selectbox("Sensibilidad al costo de:", list(catalog),
                        format_func=lambda c: catalog[c]["name"])
    costs = np.linspace(0, max(catalog[code]["cost"] * 3, 50), 25)
    gains = []
    for cost in costs:
        cat = copy.deepcopy(catalog)
        cat[code]["cost"] = float(cost)
        r = finance.decide_retention(sim, catalog=cat, high_risk=high, low_risk=low,
                                     tier_thresholds=finance.compute_tier_thresholds(sim["margin"]))
        gains.append(r["ganancia_neta_esperada"].sum())
    fig = go.Figure(go.Scatter(x=costs, y=gains, mode="lines", line=dict(color=BLUE, width=2)))
    fig.add_vline(x=catalog[code]["cost"], line_dash="dash", line_color=GRAY,
                  annotation_text="costo actual")
    fig.update_layout(title=f"Ganancia neta total vs. costo de {catalog[code]['name']}",
                      xaxis_title="Costo del incentivo ($)", yaxis_title="Ganancia neta esperada ($)")
    c2.plotly_chart(style_fig(fig), width="stretch")

    st.subheader("Matriz de decisión simulada")
    st.dataframe(finance.segment_matrix(sim).rename(columns=MATRIX_COLS).style.format(MATRIX_FMT),
                 hide_index=True, width="stretch")


# Navegación
def main():
    scored, thresholds = load_portfolio()
    with st.sidebar:
        st.markdown("## 🏦 Deep Retention")
        st.caption("Sistema de Retención Selectiva de Clientes")
        page = st.radio("Sección", ["Panel de Cartera", "Buscador de Clientes",
                                    "Simulador de Parámetros"], label_visibility="collapsed")
        st.divider()
        st.markdown("**Regla de decisión**")
        st.latex(r"G = P \times E \times M - C")
        st.caption("Solo se retiene al cliente si la ganancia neta esperada G es mayor que 0.")

    if page == "Panel de Cartera":
        page_dashboard(scored)
    elif page == "Buscador de Clientes":
        page_search(scored, thresholds)
    else:
        page_simulator(scored, thresholds)


main()

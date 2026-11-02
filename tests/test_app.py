"""Pruebas de humo de la aplicación Streamlit (sin navegador)."""
import pytest
from streamlit.testing.v1 import AppTest

from src import config
from src.portfolio import SCORED_FILE

APP_FILE = str(config.ROOT_DIR / "app" / "app.py")

pytestmark = pytest.mark.skipif(not SCORED_FILE.exists(), reason="Cartera no evaluada")


def _app():
    return AppTest.from_file(APP_FILE, default_timeout=120).run()


def _metrics(at):
    return {m.label: m.value for m in at.metric}


def test_panel_de_cartera():
    at = _app()
    assert not at.exception
    assert _metrics(at)["Clientes analizados"] == "10,000"


def test_buscador_cliente_existente():
    at = _app()
    at.sidebar.radio[0].set_value("Buscador de Clientes").run()
    at.selectbox[0].set_value(at.selectbox[0].options[0]).run()
    assert not at.exception
    assert "Probabilidad de fuga" in _metrics(at)


def test_buscador_cliente_nuevo():
    at = _app()
    at.sidebar.radio[0].set_value("Buscador de Clientes").run()
    at.radio[0].set_value("Nuevo cliente").run()
    at.button[0].click().run()
    assert not at.exception
    assert "Probabilidad de fuga" in _metrics(at)


def test_simulador_reacciona_al_costo():
    at = _app()
    at.sidebar.radio[0].set_value("Simulador de Parámetros").run()
    base = _metrics(at)["Inversión"]
    at.number_input[0].set_value(400.0).run()
    assert not at.exception
    assert _metrics(at)["Inversión"] != base

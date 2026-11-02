"""Configuración central del proyecto Deep Retention.

Todas las rutas y parámetros globales se definen aquí para que el resto
de los módulos (entrenamiento, inferencia y aplicación) usen los mismos
valores y el proyecto sea reproducible.
"""
from pathlib import Path

# --- Rutas ---
ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
DOCS_DIR = ROOT_DIR / "docs"

# --- Reproducibilidad ---
RANDOM_STATE = 42

# --- División de datos ---
TEST_SIZE = 0.15         # proporción del conjunto de prueba
VALIDATION_SIZE = 0.15   # proporción del conjunto de validación

# Los parámetros del dataset (columnas, objetivo) y los parámetros
# financieros (costos y efectividad de incentivos) se agregarán en
# fases posteriores, una vez confirmado el dataset.

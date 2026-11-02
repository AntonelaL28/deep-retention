"""Configuración central del proyecto Deep Retention.

Todas las rutas y parámetros globales se definen aquí para que el resto
de los módulos (entrenamiento, inferencia y aplicación) usen los mismos
valores y el proyecto sea reproducible.
"""
from pathlib import Path

# Rutas
ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
DOCS_DIR = ROOT_DIR / "docs"

# Reproducibilidad
RANDOM_STATE = 42

# Division de datos
TEST_SIZE = 0.15         # proporción del conjunto de prueba
VALIDATION_SIZE = 0.15   # proporción del conjunto de validación

# Dataset
RAW_DATA_FILE = RAW_DATA_DIR / "Customer-Churn-Records.csv"
TARGET = "Exited"                                   # 1 = el cliente abandonó
ID_COLUMNS = ["RowNumber", "CustomerId", "Surname"]  # no aportan información predictiva

# Columnas excluidas por fuga de datos (data leakage): se registran después
# de que el cliente abandona, por lo que revelan la respuesta.
LEAKAGE_COLUMNS = ["Complain"]

# Variables del modelo (definidas en el EDA, notebooks/01_eda.ipynb).
# Se excluyen Card Type, Satisfaction Score y Point Earned por no tener
# asociación con el abandono. NumOfProducts se trata como categórica.
# Solo se usan las que existan en el archivo.
NUMERIC_FEATURES = ["CreditScore", "Age", "Tenure", "Balance", "EstimatedSalary"]
BINARY_FEATURES = ["HasCrCard", "IsActiveMember"]
CATEGORICAL_FEATURES = ["Geography", "Gender", "NumOfProducts"]


# Parámetros financieros (lógica de retención)
# Umbrales de probabilidad de fuga (matriz de decisión comercial).
HIGH_RISK_THRESHOLD = 0.65   # >= : riesgo alto
LOW_RISK_THRESHOLD = 0.30    # <  : cliente estable, sin acción

# Catálogo de incentivos de retención.
INCENTIVE_CATALOG = {
    "GESTOR_VIP": {
        "name": "Gestor VIP y Tarifa Plana",
        "cost": 250.0,
        "effectiveness": 0.45,
        "tiers": ["Superior"],
    },
    "EXONERACION": {
        "name": "Exoneración de Comisiones",
        "cost": 80.0,
        "effectiveness": 0.30,
        "tiers": ["Superior", "Medio"],
    },
    "BONIFICACION": {
        "name": "Bonificación Transaccional",
        "cost": 40.0,
        "effectiveness": 0.25,
        "tiers": ["Superior", "Medio", "Bajo"],
    },
}

# Percentiles del saldo usados para definir los niveles de valor patrimonial.
TIER_HIGH_QUANTILE = 0.75    # saldo >= p75 -> "Superior"
TIER_LOW_QUANTILE = 0.40     # saldo >= p40 -> "Medio", si no -> "Bajo"

# Modelo (MLP)
MODEL_FILE = MODELS_DIR / "deep_retention_mlp.keras"
HIDDEN_UNITS = (128, 64)
DROPOUT_RATES = (0.3, 0.2)
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
MAX_EPOCHS = 300
EARLY_STOPPING_PATIENCE = 15

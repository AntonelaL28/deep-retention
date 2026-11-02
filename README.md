# Deep Retention — Sistema de Retención Selectiva de Clientes

Motor predictivo basado en **redes neuronales** que estima la probabilidad de abandono (churn) de los clientes de una entidad bancaria y, mediante un filtro de rentabilidad financiera neta, decide si vale la pena invertir en retenerlos.

## Problema

La retención indiscriminada de clientes reduce el margen operativo del banco: se gastan promociones y bonificaciones en clientes con saldos mínimos o que generan pérdidas. El sistema solo autoriza un plan de retención cuando el cliente representa una **ganancia potencial neta** tras descontar el costo del incentivo.

## Regla de decisión

```
Ganancia Neta Esperada = (P(fuga) × Efectividad del Incentivo × Margen del Cliente) − Costo del Incentivo
```

La acción comercial solo se ejecuta si la **Ganancia Neta Esperada > 0**. En caso contrario, el cliente se cataloga como **"Salida Permitida"**.

### Segmentación

| Segmento | Condición | Decisión |
|---|---|---|
| A: Alto Valor | Ganancia > 0 y saldo superior | Retención Prioritaria |
| B: Medio | Ganancia > 0 y saldo medio/bajo | Retención Automatizada |
| C: No Rentable | Ningún incentivo genera ganancia > 0 (incluye clientes deficitarios) | Salida Permitida |
| D: Estable | P(fuga) < 0.30 | Sin Acción Requerida |

Para probabilidades entre 0.30 y 0.65 (rango no cubierto por la matriz original) la decisión la toma directamente la fórmula de ganancia neta. Entre los incentivos permitidos para cada cliente se elige el que maximiza la ganancia.

## Componentes

| Componente | Descripción |
|---|---|
| Motor de predicción | Perceptrón Multicapa (MLP) que devuelve una probabilidad de abandono entre 0 y 1 |
| Filtro de rentabilidad | Cruza la probabilidad con el margen del cliente para decidir si se interviene |
| Catálogo de acciones | Asigna incentivos según el perfil del cliente y su costo unitario |
| Aplicación (Streamlit) | Buscador por cliente, panel de cartera y simulador de parámetros |
| Informe ejecutivo | Resultados financieros (ROI), métricas del modelo y explicabilidad (SHAP) |

## Dataset

`data/raw/Customer-Churn-Records.csv`: 10.000 clientes de un banco europeo (Francia, Alemania y España) con 18 columnas. La variable objetivo es `Exited` (1 = abandonó), con un **20,4%** de abandono.

Decisiones del análisis exploratorio ([notebooks/01_eda.ipynb](notebooks/01_eda.ipynb)):

- Se excluye `Complain` por **fuga de datos** (correlación 0,996 con `Exited`).
- Se excluyen `Card Type`, `Satisfaction Score` y `Point Earned` por no tener asociación con el abandono.
- `NumOfProducts` se trata como categórica por su relación no lineal con el abandono.

## Estructura del repositorio

```
deep-retention/
├── app/                 # Aplicación interactiva (Streamlit)
├── data/
│   ├── raw/             # Datos originales
│   └── processed/       # Datos procesados
├── docs/                # Informe ejecutivo (PDF)
├── models/              # Modelo entrenado (.keras), preprocesador e historial
├── notebooks/
│   └── 01_eda.ipynb     # Análisis exploratorio de datos
├── reports/figures/     # Gráficos generados
├── src/                 # Código fuente (preprocesamiento, entrenamiento, inferencia)
│   ├── config.py        # Rutas y parámetros globales (umbrales, catálogo de incentivos)
│   ├── finance.py       # Regla de ganancia neta, segmentación y KPIs de cartera
│   ├── preprocessing.py # Carga, división estratificada, escalado y codificación
│   ├── model.py         # Arquitectura del MLP y entrenamiento
│   ├── train.py         # Script de entrenamiento
│   └── predict.py       # Inferencia con el modelo guardado
├── tests/               # Pruebas unitarias
├── requirements.txt
└── README.md
```

## Instalación

Requiere **Python 3.10 – 3.13**.

```bash
git clone https://github.com/javier04s/deep-retention.git
cd deep-retention
python -m venv venv
```

Activar el entorno virtual:

```bash
# Windows (Git Bash)
source venv/Scripts/activate
# Windows (CMD / PowerShell)
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate
```

Instalar dependencias:

```bash
pip install -r requirements.txt
```

## Entrenamiento del modelo

```bash
python -m src.train
```

Entrena el MLP y guarda en `models/` el modelo, el preprocesador y el historial de entrenamiento. El repositorio ya incluye el modelo entrenado, por lo que este paso es opcional.

| Capa | Configuración |
|---|---|
| Entrada | 16 variables (escaladas y codificadas) |
| Oculta 1 | Dense 128 + Batch Normalization + ReLU + Dropout 0.3 |
| Oculta 2 | Dense 64 + ReLU + Dropout 0.2 |
| Salida | Dense 1 + Sigmoide (probabilidad de fuga) |

- Optimizador Adam (lr = 0.001), pérdida binary cross-entropy.
- Desbalance: `class_weight` balanceado (abandona ≈ 2.45, se queda ≈ 0.63).
- Early Stopping sobre `val_loss` con paciencia de 15 épocas.
- División 70 / 15 / 15 (entrenamiento / validación / prueba), estratificada.

## Pruebas

```bash
python -m pytest -v
```

## Informe ejecutivo

📄 *Disponible al finalizar el proyecto.*

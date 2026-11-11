| | |
|---|---|
| **Materia** | Inteligencia Artificial: Aprendizaje Automático |
| **Universidad** | Universidad Católica Andrés Bello (UCAB) |
| **Profesor** | Gustavo La Fontaine |
| **Integrantes** | Fernando Gil · Melanie Gamboa · Antonela Lauria · Javier Delmoral |

# Link del Informe Ejecutivo del Proyecto
https://docs.google.com/document/d/1xrjeh8SBMbmCZ6OFWoOKrNIxPiXe-cqX/edit?usp=sharing&ouid=104416054958940764766&rtpof=true&sd=true

# Breve descripción del Sistema de Retención Selectiva de Clientes
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
| A: Alto Valor | Ganancia > 0 y valor superior | Retención Prioritaria |
| B: Medio | Ganancia > 0 y valor medio/bajo | Retención Automatizada |
| C: No Rentable | Ningún incentivo genera ganancia > 0 (incluye clientes deficitarios) | Salida Permitida |
| D: Estable | P(fuga) < 0.30 | Sin Acción Requerida |

Para probabilidades entre 0.30 y 0.65 (rango no cubierto por la matriz original) la decisión la toma directamente la fórmula de ganancia neta. Entre los incentivos permitidos para cada cliente se elige el que maximiza la ganancia. El nivel de valor se define por el **margen anual** del cliente (no por el saldo, ya que el 36% de los clientes tiene saldo 0).

### Margen del cliente (supuestos)

```
Margen anual = 2% × Saldo + $50 × Productos + Ingreso por tarjeta + Comisiones transaccionales − $180
```

| Componente | Supuesto |
|---|---|
| Margen de interés | 2% del saldo |
| Comisiones por producto | $50 por producto |
| Tarjeta de crédito | $40 (Silver), $80 (Gold), $120 (Platinum), $160 (Diamond) |
| Comisiones transaccionales | 0,4% del salario si es activo; 0,1% si es inactivo |
| Costo de servicio | $180 por cliente al año |

### Catálogo de incentivos

| Incentivo | Costo | Efectividad | Niveles de valor |
|---|---|---|---|
| Gestor VIP y Tarifa Plana | $250 | 45% | Superior |
| Exoneración de Comisiones | $80 | 30% | Superior, Medio |
| Bonificación Transaccional | $40 | 25% | Todos (excepto deficitarios) |

Todos los supuestos están en `src/config.py` y se pueden ajustar en el simulador de la aplicación.

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
├── app/
│   └── app.py           # Aplicación interactiva (Streamlit)
├── data/
│   ├── raw/             # Datos originales
│   └── processed/       # Cartera evaluada (probabilidad, margen y decisión por cliente)
├── docs/                # Informe ejecutivo (PDF) y capturas de la aplicación
├── models/              # Modelo entrenado (.keras), preprocesador e historial
├── notebooks/
│   └── 01_eda.ipynb     # Análisis exploratorio de datos
├── reports/
│   ├── figures/         # Gráficos generados
│   ├── metrics.json     # Métricas de evaluación
│   ├── resumen_cartera.json  # KPIs financieros de la cartera
│   └── matriz_decision.csv   # Matriz de decisión por segmento
├── src/                 # Código fuente (preprocesamiento, entrenamiento, inferencia)
│   ├── config.py        # Rutas y parámetros globales (umbrales, catálogo de incentivos)
│   ├── finance.py       # Regla de ganancia neta, segmentación y KPIs de cartera
│   ├── preprocessing.py # Carga, división estratificada, escalado y codificación
│   ├── model.py         # Arquitectura del MLP y entrenamiento
│   ├── train.py         # Script de entrenamiento
│   ├── calibration.py   # Calibración de probabilidades (Platt)
│   ├── evaluate.py      # Métricas, gráficos y SHAP sobre el conjunto de prueba
│   ├── predict.py       # Inferencia con el modelo guardado
│   ├── explain.py       # Explicación SHAP de un cliente individual
│   └── portfolio.py     # Aplica el sistema completo a la cartera
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

## Reproducir desde cero

Con el entorno activado, estos comandos regeneran todos los resultados en orden:

```bash
python -m src.train        # 1. Entrena la red y la calibra       -> models/
python -m src.evaluate     # 2. Métricas, gráficos y SHAP          -> reports/
python -m src.portfolio    # 3. Evalúa la cartera completa         -> data/processed/, reports/
streamlit run app/app.py   # 4. Abre la aplicación
```

El repositorio ya incluye el modelo entrenado y los resultados, por lo que se puede ir directo al paso 4. Volver a entrenar puede producir variaciones mínimas en las métricas según el sistema operativo.

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

## Evaluación

```bash
python -m src.evaluate
```

Resultados sobre el conjunto de prueba (1.500 clientes no vistos en el entrenamiento):

| Métrica | Valor |
|---|---|
| ROC-AUC | 0.866 |
| PR-AUC | 0.701 (azar: 0.204) |
| Brier Score | 0.137 → **0.103** (sin calibrar → calibrado) |
| Error de calibración (ECE) | 0.158 → **0.022** |

**Calibración:** `class_weight` hace que la red sobreestime la probabilidad de fuga (promedio predicho 36% frente a 20,4% real). Como la fórmula de ganancia neta usa esa probabilidad directamente, se aplica calibración de Platt ajustada con el conjunto de validación.

**Factores de abandono (SHAP):** número de productos, edad y actividad del cliente, seguidos de género, saldo y país.

## Resultados sobre la cartera

```bash
python -m src.portfolio
```

| Indicador | Valor |
|---|---|
| Clientes analizados | 10.000 |
| Clientes en riesgo (P ≥ 0.30) | 2.240 |
| Clientes a intervenir (A + B) | 1.782 |
| Salidas permitidas (C) | 458 |
| Capital en riesgo crítico (P ≥ 0.65) | $64,1 M |
| Inversión en retención | $205.950 |
| Ganancia neta esperada | $653.446 |
| **ROI** | **317%** |

Frente a retener a todos los clientes en riesgo sin filtro de rentabilidad, el sistema evita invertir $18.320 en clientes no rentables y $13.615 de pérdida esperada (ROI 285% → 317%).

## Aplicación

```bash
streamlit run app/app.py
```

Se abre en el navegador (http://localhost:8501) con tres secciones:

| Sección | Qué permite |
|---|---|
| **Panel de Cartera** | KPIs de la cartera (capital en riesgo, capital rescatable, inversión, ganancia neta y ROI), gráficos de riesgo y valor, matriz de decisión y lista descargable de clientes a contactar. Filtros por país, nivel de valor y decisión. |
| **Buscador de Clientes** | Búsqueda por ID o apellido, o evaluación de un **cliente nuevo**: probabilidad de fuga, margen, decisión sugerida con el cálculo de la ganancia neta y los factores SHAP que explican el riesgo. |
| **Simulador de Parámetros** | Ajuste de costos y efectividad de cada incentivo, umbrales de riesgo y supuestos de margen; compara contra el escenario base y muestra la sensibilidad de la ganancia al costo de cada incentivo. |

![Panel de Cartera](docs/img/app_panel.png)
![Buscador de Clientes](docs/img/app_buscador.png)
![Simulador de Parámetros](docs/img/app_simulador.png)

## Pruebas

```bash
python -m pytest -v
```

## Informe ejecutivo

📄 *Disponible al finalizar el proyecto.*

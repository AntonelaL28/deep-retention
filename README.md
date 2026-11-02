# Deep Retention — Sistema de Retención Selectiva de Clientes

Motor predictivo basado en **redes neuronales** que estima la probabilidad de abandono (churn) de los clientes de una entidad bancaria y, mediante un filtro de rentabilidad financiera neta, decide si vale la pena invertir en retenerlos.

## Problema

La retención indiscriminada de clientes reduce el margen operativo del banco: se gastan promociones y bonificaciones en clientes con saldos mínimos o que generan pérdidas. El sistema solo autoriza un plan de retención cuando el cliente representa una **ganancia potencial neta** tras descontar el costo del incentivo.

## Regla de decisión

```
Ganancia Neta Esperada = (P(fuga) × Efectividad del Incentivo × Margen del Cliente) − Costo del Incentivo
```

La acción comercial solo se ejecuta si la **Ganancia Neta Esperada > 0**. En caso contrario, el cliente se cataloga como **"Salida Permitida"**.

## Componentes

| Componente | Descripción |
|---|---|
| Motor de predicción | Perceptrón Multicapa (MLP) que devuelve una probabilidad de abandono entre 0 y 1 |
| Filtro de rentabilidad | Cruza la probabilidad con el margen del cliente para decidir si se interviene |
| Catálogo de acciones | Asigna incentivos según el perfil del cliente y su costo unitario |
| Aplicación (Streamlit) | Buscador por cliente, panel de cartera y simulador de parámetros |
| Informe ejecutivo | Resultados financieros (ROI), métricas del modelo y explicabilidad (SHAP) |

## Estructura del repositorio

```
deep-retention/
├── app/                 # Aplicación interactiva (Streamlit)
├── data/
│   ├── raw/             # Datos originales
│   └── processed/       # Datos procesados
├── docs/                # Informe ejecutivo (PDF)
├── models/              # Modelo entrenado y preprocesador
├── notebooks/           # Análisis exploratorio y experimentos
├── reports/figures/     # Gráficos generados
├── src/                 # Código fuente (preprocesamiento, entrenamiento, inferencia)
├── requirements.txt
└── README.md
```

## Instalación

Requiere **Python 3.10 – 3.12**.

```bash
git clone <url-del-repositorio>
cd deep-retention
python -m venv venv
# Windows: venv\Scripts\activate | Linux/Mac: source venv/bin/activate
pip install -r requirements.txt
```

## Informe ejecutivo

📄 *Disponible al finalizar el proyecto.*

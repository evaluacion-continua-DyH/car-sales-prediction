#  Car Sales Price Prediction — Machine Learning Project

##  Introducción
El objetivo de este proyecto es desarrollar un modelo de Machine Learning capaz de **predecir el precio de un coche** en función de varias características técnicas y de mercado.

Este trabajo forma parte del módulo de *Herramientas de Trabajo Colaborativo*, donde se han aplicado buenas prácticas de desarrollo colaborativo utilizando Git, GitHub y un flujo profesional basado en ramas, commits y Pull Requests.

El proyecto incluye:
- Preprocesamiento del dataset  
- Entrenamiento de un modelo **XGBoostRegressor**  
- Evaluación de métricas  
- Pipeline reproducible  
- Script de predicción para inferencia

---

##  Problema a resolver

Dado un conjunto de características de un coche (motor, año, fabricante, modelo, tipo de combustible, etc.), queremos estimar su precio en el mercado de segunda mano.

Formalmente, buscamos una función:

\[
\hat{y} = f(X)
\]

donde:

- \(X\) es el vector de características del vehículo  
- \(\hat{y}\) es el precio predicho  

El objetivo es minimizar el error absoluto medio:

\[
MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
\]

---

##  Descripción del Dataset

El dataset utilizado se encuentra en:

```
data/car_sales_dataset.csv
```

Contiene información sobre coches vendidos con las siguientes columnas:

| Columna                | Tipo        | Descripción |
|-----------------------|-------------|-------------|
| Manufacturer          | Categórica  | Marca del vehículo |
| Model                 | Categórica  | Modelo del coche |
| Engine size           | Numérica    | Tamaño del motor (litros) |
| Year of manufacture   | Numérica    | Año de fabricación |
| Mileage               | Numérica    | Kilometraje |
| Fuel type             | Categórica  | Tipo de combustible |
| Price                 | Numérica    | **Variable objetivo** |

###  Limpieza y valores nulos
El dataset contenía valores faltantes y algunas inconsistencias menores que fueron tratadas en el pipeline de preprocesamiento.

---

##  Descripción del Pipeline

El proyecto utiliza un pipeline modular dividido en dos scripts:

- `preprocesamiento.py`
- `training.py`

A continuación se muestra el diagrama que representa el flujo:

```mermaid
flowchart TD
    A[Raw Dataset
car_sales_dataset.csv] --> B[Preprocesamiento]
    B --> C[Imputación de valores
(Num: median, Cat: most_frequent)]
    C --> D[Escalado
StandardScaler]
    C --> E[Codificación
OneHotEncoder]
    D --> F[Dataset Procesado
car_sales_processed.csv]
    E --> F

    F --> G[Train/Test Split]
    G --> H[XGBoostRegressor]
    H --> I[Modelo Entrenado
xgb_model.pkl]

    B --> J[Preprocessor.pkl]
```

###  Detalles del pipeline

#### 1. Columnas numéricas:
- `Engine size`
- `Year of manufacture`
- `Mileage`

Procesamiento:
- `SimpleImputer(strategy="median")`
- `StandardScaler()`

#### 2. Columnas categóricas:
- `Manufacturer`
- `Model`
- `Fuel type`

Procesamiento:
- `SimpleImputer(strategy="most_frequent")`
- `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`

---

## 🤖 Modelo utilizado

El modelo final seleccionado fue:

```
XGBoostRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
```

Este modelo presenta un excelente rendimiento para datos tabulares con relaciones no lineales.

---

##  Resultados obtenidos

Tras entrenar el modelo y evaluarlo sobre un 20% de los datos se obtuvieron las siguientes métricas:

| Métrica | Valor |
|--------|-------|
| **MAE** | 186.97 |
| **RMSE** | 311.25 |
| **R²** | 0.9996 |

###  Interpretación
- El modelo predice precios con un error promedio de **187 €**, extremadamente bajo.
- Con un **R² de 0.9996**, el modelo explica prácticamente toda la variabilidad del precio.
- El RMSE indica una desviación media muy pequeña entre predicción y valor real.

---

##  Scripts incluidos

| Script | Descripción |
|--------|-------------|
| `preprocesamiento.py` | Construye el pipeline de preprocesamiento y guarda dataset procesado + preprocessor.pkl |
| `training.py` | Entrena el modelo XGBoost y guarda xgb_model.pkl |
| `prediction.py` | Genera predicciones usando modelo + preprocesador |

---

##  Conclusión

El proyecto demuestra el proceso completo de construcción de un sistema de Machine Learning real:

1. Procesamiento y transformación del dataset  
2. Entrenamiento de un modelo avanzado  
3. Evaluación exhaustiva  
4. Pipeline reproducible  
5. Scripts automatizados para futuras predicciones  

Este flujo permite integrarlo fácilmente en APIs, dashboards o procesos batch.

---

##  Autores
Proyecto desarrollado por Diego Mosquera y Hernando de las Bárcenas


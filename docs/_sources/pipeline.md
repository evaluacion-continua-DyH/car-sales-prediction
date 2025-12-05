# Pipeline Completo del Proyecto

El proyecto **Car Sales Prediction** utiliza un pipeline profesional basado en dos fases principales:

1. **Preprocesamiento del dataset**
2. **Entrenamiento del modelo XGBoost**

Ambas fases se implementan mediante scripts reproducibles (`preprocesamiento.py` y `training.py`) y se documentan en este Jupyter Book.

---

# 🧩 1. Preprocesamiento del dataset

El preprocesamiento se documenta de manera completa en el notebook:

notebooks/preprocesamiento.ipynb


y se implementa en el script:

src/preprocesamiento.py


## 🔧 Técnicas utilizadas

### 🔹 Variables numéricas
- Imputación con la **mediana**
- Escalado mediante **StandardScaler**

### 🔹 Variables categóricas
- Imputación con el valor **más frecuente**
- **OneHotEncoding** con `handle_unknown="ignore"`

## 🔨 Construcción del preprocesador

El pipeline se implementa usando `ColumnTransformer`:

```python
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, num_cols),
        ("cat", categorical_pipeline, cat_cols)
    ]
)

El resultado del preprocesamiento se guarda en:
data/processed/car_sales_processed.csv
data/processed/preprocessor.pkl

El entrenamiento se gestiona desde:
src/training.py



El modelo principal entrenado es:
model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

flowchart TD
    A[Dataset Original<br>car_sales_dataset.csv] --> B[Preprocesamiento<br>ColumnTransformer]
    B --> C[Dataset Procesado<br>car_sales_processed.csv]
    C --> D[Train/Test Split<br>80% / 20%]
    D --> E[XGBoost Training<br>500 árboles]
    E --> F[Evaluación<br>MAE, RMSE, R²]
    F --> G[Guardado del Modelo<br>xgb_model.pkl]
    B --> H[Guardado del Preprocesador<br>preprocessor.pkl]

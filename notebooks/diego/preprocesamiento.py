

import os
import pandas as pd
import numpy as np
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


# ============================================
# 1. Rutas
# ============================================

# ============================================
# 1. Rutas robustas
# ============================================

# Carpeta del script actual (notebooks/diego/)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Subimos dos niveles hasta car-sales-prediction/
repo_root = os.path.dirname(os.path.dirname(current_dir))

# Dataset
raw_path = os.path.join(repo_root, "data", "car_sales_dataset.csv")

# Directorio de salida
processed_dir = os.path.join(repo_root, "data", "processed")
os.makedirs(processed_dir, exist_ok=True)

output_data_path = os.path.join(processed_dir, "car_sales_processed.csv")
output_preprocessor_path = os.path.join(processed_dir, "preprocessor.pkl")

print("Ruta del proyecto:", repo_root)
print("Buscando dataset en:", raw_path)

# ============================================
# 2. Carga del dataset
# ============================================

print(f"Cargando dataset desde: {raw_path}")
df = pd.read_csv(raw_path)


# ============================================
# 3. Separación de columnas
# ============================================

target = "Price"  # variable objetivo, aunque aquí no la usamos para entrenar

num_cols = ["Engine size", "Year of manufacture", "Mileage"]
cat_cols = ["Manufacturer", "Model", "Fuel type"]

print("Columnas numéricas:", num_cols)
print("Columnas categóricas:", cat_cols)


# ============================================
# 4. Pipeline de preprocesamiento
# ============================================

numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, num_cols),
        ("cat", categorical_pipeline, cat_cols)
    ]
)


# ============================================
# 5. Aplicación del preprocesamiento
# ============================================

X = df.drop(columns=[target])
y = df[target]

print("Aplicando preprocesamiento...")

X_processed = preprocessor.fit_transform(X)

# Obtener nombres de columnas generadas
processed_cols = (
    preprocessor.named_transformers_["num"]
        .get_feature_names_out(num_cols).tolist()
    +
    preprocessor.named_transformers_["cat"]
        .named_steps["onehot"]
        .get_feature_names_out(cat_cols).tolist()
)

# Reconstruir dataframe procesado
df_processed = pd.DataFrame(X_processed, columns=processed_cols)
df_processed[target] = y.values


# ============================================
# 6. Guardar resultados
# ============================================

df_processed.to_csv(output_data_path, index=False)
joblib.dump(preprocessor, output_preprocessor_path)

print(f"✔ Dataset procesado guardado en: {output_data_path}")
print(f"✔ Preprocesador guardado en: {output_preprocessor_path}")
print("Preprocesamiento completado.")

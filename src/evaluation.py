# ============================================
# EVALUACIÓN Y VALIDACIÓN DEL MODELO XGBOOST
# ============================================

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ============================================
# 1. Rutas robustas
# ============================================

current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(current_dir)

model_path = os.path.join(repo_root, "models", "xgb_model.pkl")
preprocessor_path = os.path.join(repo_root, "models", "preprocessor.pkl")
data_path = os.path.join(repo_root, "data", "processed", "car_sales_processed.csv")

print("Model:", model_path)
print("Preprocessor:", preprocessor_path)
print("Dataset:", data_path)

# ============================================
# 2. Cargar modelo, preprocesador y dataset
# ============================================

print("\nCargando modelo y datos...")
model = joblib.load(model_path)
preprocessor = joblib.load(preprocessor_path)

df = pd.read_csv(data_path)

print("✔ Recursos cargados correctamente.")

# Separar variables
TARGET = "Price"
X = df.drop(columns=[TARGET])
y = df[TARGET]

# ============================================
# 3. División Train/Test (idéntica a training.py)
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============================================
# 4. Preprocesar y evaluar
# ============================================

print("\nProcesando datos...")
X_test_processed = preprocessor.transform(X_test)

print("Generando predicciones...")
y_pred = model.predict(X_test_processed)

# Métricas
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# ============================================
# 5. Mostrar resultados
# ============================================

print("\n=========== RESULTADOS DE EVALUACIÓN ===========")
print(f"MAE : {mae:,.2f}")
print(f"RMSE: {rmse:,.2f}")
print(f"R²  : {r2:.4f}")
print("================================================\n")


# ============================================
# 6. Función pública por si este módulo se importa
# ============================================

def evaluate_model():
    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "y_true": y_test.values,
        "y_pred": y_pred
    }


if __name__ == "__main__":
    evaluate_model()

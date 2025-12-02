# ============================================
# ENTRENAMIENTO DEL MODELO XGBOOST
# ============================================

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import joblib
import pandas as pd
import os

# ----------------------------------------------------------
# 1. Cargar dataset procesado
# ----------------------------------------------------------
# ============================================
# Rutas robustas garantizadas
# ============================================

current_dir = os.path.dirname(os.path.abspath(__file__))

# Subimos un nivel (src → project root)
repo_root = os.path.dirname(current_dir)

print("current_dir:", current_dir)
print("repo_root:", repo_root)

# Rutas correctas
processed_dir = os.path.join(repo_root, "data", "processed")
model_dir = os.path.join(repo_root, "models")

data_path = os.path.join(processed_dir, "car_sales_processed.csv")
preprocessor_path = os.path.join(processed_dir, "preprocessor.pkl")

print("Buscando CSV en:", data_path)

df = pd.read_csv(data_path)

TARGET = "Price"
X = df.drop(columns=[TARGET])
y = df[TARGET]

# ----------------------------------------------------------
# 2. División Train/Test
# ----------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train:", X_train.shape, "| Test:", X_test.shape)

# ----------------------------------------------------------
# 3. Definir modelo XGBoost
# ----------------------------------------------------------

model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# ----------------------------------------------------------
# 4. Entrenar modelo
# ----------------------------------------------------------

print("Entrenando modelo XGBoost...")
model.fit(X_train, y_train)

# ----------------------------------------------------------
# 5. Evaluación
# ----------------------------------------------------------

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n======= MÉTRICAS DEL MODELO =======")
print(f"MAE:  {mae:,.2f}")
print(f"RMSE: {rmse:,.2f}")
print(f"R²:   {r2:.4f}")
print("===================================\n")

# ----------------------------------------------------------
# 6. Guardar modelo y preprocesador
# ----------------------------------------------------------

model_output_path = os.path.join(model_dir, "xgb_model.pkl")
preprocessor_output_path = os.path.join(model_dir, "preprocessor.pkl")

joblib.dump(model, model_output_path)
joblib.dump(joblib.load(preprocessor_path), preprocessor_output_path)

print(f"✔ Modelo guardado en: {model_output_path}")
print(f"✔ Preprocesador copiado a: {preprocessor_output_path}")
print("Entrenamiento completado.")

# ============================================
# PREDICCIÓN DEL MODELO XGBOOST
# ============================================

import joblib
import pandas as pd
import numpy as np
import os

# ============================================
# 1. Rutas robustas
# ============================================

current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(current_dir)

model_dir = os.path.join(repo_root, "models")
data_dir = os.path.join(repo_root, "data", "raw")  # si quieres cargar datos sin procesar

model_path = os.path.join(model_dir, "xgb_model.pkl")
preprocessor_path = os.path.join(model_dir, "preprocessor.pkl")

print("Model path:", model_path)
print("Preprocessor path:", preprocessor_path)

# ============================================
# 2. Cargar modelo y preprocesador
# ============================================

print("Cargando modelo y preprocesador...")
model = joblib.load(model_path)
preprocessor = joblib.load(preprocessor_path)

print("✔ Modelo y preprocesador cargados correctamente.")

# ============================================
# 3. Función para realizar predicciones
# ============================================

def make_prediction(input_data):
    """
    Acepta:
        - Un diccionario con los datos de un coche
        - Un DataFrame
    Devuelve:
        - Predicciones en forma de array
    """

    # Convertir a DataFrame si es un dict
    if isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
    elif isinstance(input_data, pd.DataFrame):
        df = input_data.copy()
    else:
        raise ValueError("input_data debe ser un dict o un DataFrame")

    # Preprocesar datos
    processed = preprocessor.transform(df)

    # Predecir
    prediction = model.predict(processed)

    return prediction


# ============================================
# 4. Ejemplo de uso (puedes borrar esto si quieres)
# ============================================

if __name__ == "__main__":

    ejemplo = {
        "Year": 2018,
        "Mileage": 35000,
        "Brand": "Toyota",
        "Model": "Corolla",
        "Fuel_Type": "Gasoline",
        "Transmission": "Automatic",
        "Engine_Size": 1.8,
    }

    print("\nEjemplo de predicción:")
    pred = make_prediction(ejemplo)
    print(f"Precio estimado: {pred[0]:,.2f} €")

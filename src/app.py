import streamlit as st
import numpy as np
import pandas as pd

st.title("Car Sales Price Prediction")

st.write(
    "Aplicación de demostración para simular la predicción del precio de un coche "
    "a partir de algunas características básicas."
)

# ---- Entrada de datos del usuario ----
st.sidebar.header("Características del vehículo")

year = st.sidebar.number_input(
    "Año de matriculación",
    min_value=1990,
    max_value=2025,
    value=2015,
    step=1,
)

kms = st.sidebar.number_input(
    "Kilómetros recorridos",
    min_value=0,
    max_value=500_000,
    value=60_000,
    step=1_000,
)

engine_power = st.sidebar.number_input(
    "Potencia (CV)",
    min_value=50,
    max_value=500,
    value=120,
    step=5,
)

fuel_type = st.sidebar.selectbox(
    "Tipo de combustible",
    ["Gasolina", "Diésel", "Híbrido", "Eléctrico"],
)

transmission = st.sidebar.selectbox(
    "Transmisión",
    ["Manual", "Automática"],
)

st.sidebar.markdown("---")

# ---- Botón de predicción (simulada) ----
if st.sidebar.button("Simular predicción de precio"):
    current_year = 2025
    age = current_year - year

    # Fórmula ficticia solo para la demo
    base_price = 30000
    price = base_price - age * 800 - kms * 0.05 + engine_power * 20
    price = max(price, 500)  # evitar valores negativos

    st.subheader("Resultado de la predicción (simulada)")
    st.success(f"Precio estimado: {price:,.0f} €")

    st.write(
        "⚠️ Esta predicción es solo una simulación para la práctica. "
        "En un sistema real, aquí se usaría el modelo entrenado."
    )

# ---- Sección de métricas históricas (simuladas) ----
st.subheader("Métricas históricas del modelo (simuladas)")

epochs = np.arange(1, 11)
rmse = np.linspace(6000, 3000, 10) + np.random.randint(-500, 500, size=10)
mae = np.linspace(4000, 1500, 10) + np.random.randint(-300, 300, size=10)

metrics_df = pd.DataFrame(
    {
        "epoch": epochs,
        "RMSE": rmse,
        "MAE": mae,
    }
).set_index("epoch")

st.line_chart(metrics_df)

st.caption("Gráfico de RMSE y MAE por época (valores ficticios para la práctica).")

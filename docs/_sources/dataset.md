# Descripción del Dataset

El dataset utilizado en este proyecto contiene información estructurada sobre vehículos en venta,
incluyendo características técnicas, atributos categóricos y el precio objetivo que se desea predecir.

Este análisis es fundamental para comprender cómo deben tratarse los datos antes de desarrollar el modelo.

---

## 📊 Estructura del dataset

El dataset original contiene las siguientes columnas principales:

| Columna               | Tipo        | Descripción |
|----------------------|-------------|-------------|
| **Engine size**      | Numérica    | Tamaño del motor en litros |
| **Year of manufacture** | Numérica | Año de fabricación del vehículo |
| **Mileage**          | Numérica    | Kilometraje recorrido |
| **Manufacturer**     | Categórica  | Marca del vehículo |
| **Model**            | Categórica  | Modelo del vehículo |
| **Fuel type**        | Categórica  | Tipo de combustible |
| **Price**            | Numérica (objetivo) | Precio de venta del coche |

---

## 📈 Objetivo del análisis

El objetivo final es predecir el **Price**, a partir del resto de variables.

Para ello, el preprocesamiento se centra en:

- Imputar valores faltantes  
- Estandarizar variables numéricas  
- Codificar variables categóricas  
- Preparar el dataset para un modelo supervisado  

---

## 📌 Consideraciones importantes

- Las variables numéricas presentan escalas diferentes, lo que justifica el uso de `StandardScaler`.
- Las variables categóricas pueden tener valores desconocidos, por lo que se utiliza `OneHotEncoder(handle_unknown="ignore")`.
- El dataset final debe conservar la variable objetivo `Price`.

Esta comprensión inicial permite construir un pipeline de preprocesamiento sólido y reproducible.

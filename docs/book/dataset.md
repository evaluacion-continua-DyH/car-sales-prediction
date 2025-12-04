# Descripción del Dataset

El dataset utilizado en este proyecto corresponde a un conjunto de información sobre vehículos y sus precios de venta.  
Su objetivo es permitir la construcción de un modelo de Machine Learning capaz de **predecir el precio de un coche** en función de sus características.

---

## 1. Contenido del Dataset

El conjunto de datos contiene las siguientes categorías de información:

- **Especificaciones del vehículo** (marca, modelo, tipo, año)
- **Prestaciones técnicas** (potencia, caballos de fuerza, número de puertas)
- **Características visuales** (color interior y exterior)
- **Información financiera** (precio de venta)
- **Atributos de mercado** (tipo de transmisión, tipo de combustible, estilo de carrocería)

---

## 2. Estructura del Dataset

A continuación se presenta una descripción resumida de las principales columnas del dataset original:

| Columna              | Descripción                                                                 |
|----------------------|-----------------------------------------------------------------------------|
| `Make`               | Marca del vehículo                                                          |
| `Model`              | Modelo del vehículo                                                         |
| `Year`               | Año de fabricación                                                          |
| `Engine HP`          | Potencia del motor en caballos                                              |
| `Engine Cylinders`   | Número de cilindros                                                         |
| `Transmission Type`  | Tipo de transmisión                                                         |
| `Driven Wheels`      | Ruedas motrices                                                             |
| `Number of Doors`    | Número de puertas                                                            |
| `Market Category`    | Segmento de mercado                                                         |
| `Vehicle Size`       | Tamaño del vehículo                                                         |
| `highway MPG`        | Consumo en carretera                                                        |
| `city MPG`           | Consumo en ciudad                                                           |
| `MSRP`               | Precio del vehículo (variable objetivo)                                     |

---

## 3. Valores Faltantes y Calidad del Dataset

Antes del preprocesamiento, el dataset presentaba:

- valores nulos en `Engine HP`
- valores nulos en `Number of Doors`
- presencia de filas duplicadas
- columnas categóricas sin codificar
- escalas numéricas muy diferentes entre sí

Estas características hacían necesario aplicar un pipeline de preprocesamiento para garantizar la correcta construcción del modelo predictivo.

---

## 4. Tamaño Final del Dataset Procesado

Tras la limpieza y transformación de los datos, se obtiene un dataset:

- sin nulos  
- sin duplicados  
- con todas las variables listas para ser utilizadas en modelos supervisados  
- completamente numérico gracias al One-Hot Encoding  
- estandarizado mediante `StandardScaler`  

Este dataset se guarda como:

data/processed/car_sales_processed.csv


y es utilizado directamente en la fase de modelado.

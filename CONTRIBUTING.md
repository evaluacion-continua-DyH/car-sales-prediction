# Guía de contribución

Gracias por colaborar en el proyecto **Car Sales Price Prediction** 👋

## Flujo de trabajo con Git

1. Crea una Issue describiendo la tarea.
2. Crea una rama desde `main` con un nombre descriptivo, por ejemplo:
   - `feature/descripcion-corta`
   - `fix/arreglar-bug-x`
3. Realiza commits pequeños y con mensajes claros, en imperativo:
   - `feat: añadir script de entrenamiento`
   - `fix: corregir lectura de datos`
4. Abre un Pull Request a `main` y asígnalo a otro miembro del equipo para revisión.
5. No hagas push directo a `main`. Todos los cambios deben ir mediante Pull Request.

## Estilo de código

- Usar Python 3.10.
- Seguir buenas prácticas de legibilidad (PEP8).
- Evitar código duplicado y dejar comentarios solo cuando sean necesarios.
- Mantener los notebooks limpios: ejecutar de principio a fin, sin celdas huérfanas.

## Estructura del proyecto

- `src/`: código fuente (entrenamiento, predicción, utilidades).
- `notebooks/`: notebooks de experimentación.
- `models/`: modelos entrenados y preprocesadores.
- `docs/`: documentación y Jupyter Book.
- `data/`: datos (o rutas a los mismos).

## Issues y Pull Requests

- Cada cambio relevante debe tener una Issue asociada.
- En el Pull Request, describir:
  - Qué se ha cambiado.
  - Cómo se ha probado.
  - Si afecta a entrenamiento, predicción o documentación.

Gracias por seguir estas normas. Facilitan el trabajo en equipo y la corrección de la práctica. 🙌
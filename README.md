# Housing AI Project — House Price Prediction (Training + Inference)

Este repositorio contiene un pequeño sistema de Inteligencia Artificial capaz de **predecir el precio de una vivienda** a partir de sus características (tamaño medio de habitaciones, población de la zona, antigüedad, localización, etc.).

El modelo se entrena utilizando el dataset **California Housing**, un conjunto de datos real utilizado habitualmente en aprendizaje automático que recoge información demográfica y de vivienda de distintos distritos de California.

A diferencia de otros ejemplos simples, este proyecto no solo entrena un modelo, sino que también permite **utilizarlo posteriormente para realizar predicciones sobre viviendas nuevas**. Es decir, simula el funcionamiento básico de un sistema predictivo real.

---

## Requisitos previos

Debes tener instalado:

- Python 3.10 o superior
- Una terminal (PowerShell, CMD, Terminal de Linux o Mac)
- Conexión a internet (solo la primera vez)

Para comprobar tu versión de Python:

```bash
python --version
```

---

## Instalación y ejecución

### 1. Clonar o descomprimir el proyecto

Coloca la carpeta del proyecto en tu equipo y abre una terminal en su interior.

### 2. Crear el entorno virtual

```bash
python3 -m venv .venv
```

### 3. Activar el entorno virtual

**macOS / Linux:**
```bash
source .venv/bin/activate
```

**Windows:**
```bash
.venv\Scripts\activate
```

> Una vez activo, verás `(.venv)` al inicio de la línea de la terminal.

### 4. Instalar las dependencias

```bash
pip install -r requirements.txt
```

### 5. Entrenar el modelo

```bash
python train.py
```

Al finalizar mostrará las métricas de evaluación (`MAE` y `RMSE`) y guardará el modelo entrenado en `outputs/`.

### 6. Ejecutar predicciones

```bash
python predict.py
```

Las predicciones se guardarán en `outputs/predictions/predictions.csv`.

---

## Estructura del proyecto

```
Proyecto2/
├── config/          # Parámetros de configuración del modelo
├── data/            # Dataset de entrada
├── outputs/         # Modelos guardados y predicciones generadas
├── src/             # Código fuente (preprocesado, entrenamiento, evaluación...)
├── train.py         # Script principal de entrenamiento
├── predict.py       # Script de inferencia / predicción
└── requirements.txt # Dependencias del proyecto
```

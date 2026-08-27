# 🔍 Anomaly Detection API

API de detección de anomalías en imágenes industriales, construida sobre **PatchCore** ([anomalib](https://github.com/open-edge-platform/anomalib)) y entrenada sobre el dataset **MVTec AD** (categoría `bottle`). Incluye una API REST con FastAPI, un frontend en Streamlit, tests automatizados y despliegue vía Docker.

## ✨ Características

- Clasificación binaria (normal / anómalo) con puntuación de anomalía (`score`)
- Visualización de **mapa de calor** (heatmap) sobre la imagen original
- Visualización de **contorno** del defecto detectado
- Validación de entrada (tipo de archivo, imágenes corruptas)
- Tests automatizados con `pytest`
- Contenedorizado con Docker
- Frontend sencillo en Streamlit para uso sin necesidad de programar

## 🏗️ Arquitectura

```
[Streamlit UI] --HTTP--> [FastAPI /predict] --> [PatchCore (anomalib)] --> [Respuesta JSON: score, contorno, heatmap]
```

- **Modelo**: PatchCore, entrenado sobre MVTec AD (`bottle`), cargado desde un checkpoint de PyTorch Lightning (`.ckpt`)
- **API**: FastAPI + Uvicorn, con el modelo cargado una única vez al arrancar (patrón `lifespan`)
- **Inferencia**: `anomalib.engine.Engine.predict()`
- **Frontend**: Streamlit, consumiendo la API vía `requests`

## 📁 Estructura del proyecto

```
.
├── main.py                # API FastAPI
├── app_streamlit.py        # Frontend Streamlit
├── test_main.py            # Tests automatizados (pytest)
├── requirements.txt        # Dependencias
├── Dockerfile
├── .dockerignore
└── model/
    └── bottle_lightning.ckpt   # Checkpoint del modelo entrenado
```

## 🚀 Instalación y uso

### Opción A: local

```bash
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt

uvicorn main:app --reload
```

La API estará disponible en `http://127.0.0.1:8000`, con documentación interactiva en `http://127.0.0.1:8000/docs`.

### Opción B: Docker

```bash
docker build -t anomaly-api .
docker run -p 8000:8000 anomaly-api
```

### Frontend (Streamlit)

Con la API corriendo (local o en Docker):

```bash
pip install streamlit requests
streamlit run app_streamlit.py
```

Se abrirá en `http://localhost:8501`.

## 📡 Endpoints

| Método | Ruta | Descripción |
|---|---|---|
| `GET` | `/` | Estado de la API |
| `POST` | `/predict` | Recibe una imagen, devuelve JSON con `score`, `is_anomalous` y las visualizaciones (contorno y heatmap) en base64 |
| `POST` | `/predict/preview` | Recibe una imagen, devuelve directamente un PNG con contorno y heatmap lado a lado (pensado para pruebas visuales desde `/docs`) |

### Ejemplo de respuesta de `/predict`

```json
{
  "filename": "005.png",
  "is_anomalous": true,
  "score": 0.7276654243469238,
  "contour_image_base64": "iVBORw0KGgo...",
  "heatmap_image_base64": "iVBORw0KGgo..."
}
```

## 🧪 Tests

```bash
pytest test_main.py -v
```

Cubre: estado de la API, predicción con imagen válida, rechazo de tipos de archivo no soportados y rechazo de imágenes corruptas.

## 🔒 Notas de seguridad

El checkpoint se carga con `weights_only=True` (protección por defecto de PyTorch ≥2.6 contra deserialización insegura), permitiendo explícitamente solo los tipos necesarios de `anomalib` vía `torch.serialization.add_safe_globals(...)`.

## 🛠️ Stack técnico

Python · FastAPI · Uvicorn · anomalib (PatchCore) · PyTorch Lightning · OpenCV · Pillow · Streamlit · pytest · Docker
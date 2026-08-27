from contextlib import asynccontextmanager
import base64
import os
import io
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
from anomalib.models import Patchcore
from anomalib.engine import Engine
from anomalib.data import PredictDataset
from PIL import Image, UnidentifiedImageError
from anomalib import PrecisionType
import torch

# Para poder cargar el modelo con wheights_only = True tenemos que añadir "PrecisionType" a la White List
torch.serialization.add_safe_globals([PrecisionType])

CHECKPOINT_PATH = "./model/bottle_lightning.ckpt"

ALLOWED_CONTENT_TYPES = {"image/png", "image/jpeg", "image/jpg"}

class PredictionResponse(BaseModel):
    filename: str = Field(..., description="Nombre del archivo de imagen recibido")
    is_anomalous: bool = Field(..., description="True si se ha detectado una anomalia")
    score: float = Field(..., description="Puntuación de anomalía (0-1 aprox.). Cuanto más alta, más se aleja de lo 'normal'")
    contour_image_base64: str = Field(..., description="Imagen original con el contorno del defecto dibujado, codificada en base64 (PNG)")
    heatmap_image_base64: str = Field(..., description="Mapa de calor de anomalía superpuesto a la imagen original, codificado en base64 (PNG)")


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Cargando modelo exportado...")
    model = Patchcore.load_from_checkpoint(CHECKPOINT_PATH, weights_only=True)
    model.eval()
    app.state.model = model
    app.state.engine = Engine(enable_progress_bar=False)
    print("Modelo listo.")
    yield
    print("Apagar API...")


app = FastAPI(title="Anomaly Detection API", lifespan=lifespan)


@app.get("/")
def read_root():
    return {"status": "API Funciona"}


def build_heatmap_overlay(original_image_path: str, anomaly_map) -> np.ndarray:
    amap = anomaly_map.squeeze().cpu().numpy()
    amap_norm = (amap - amap.min()) / (amap.max() - amap.min() + 1e-8)
    amap_uint8 = (amap_norm * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(amap_uint8, cv2.COLORMAP_JET)

    h, w = amap.shape
    original = Image.open(original_image_path).convert("RGB").resize((w, h))
    original_bgr = cv2.cvtColor(np.array(original), cv2.COLOR_RGB2BGR)

    overlay = cv2.addWeighted(original_bgr, 0.6, heatmap_color, 0.4, 0)
    return overlay


def build_contour_overlay(original_image_path: str, pred_mask) -> np.ndarray:
    mask = pred_mask.squeeze().cpu().numpy()
    mask_uint8 = (mask.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = mask.shape
    original = Image.open(original_image_path).convert("RGB").resize((w, h))
    original_bgr = cv2.cvtColor(np.array(original), cv2.COLOR_RGB2BGR)

    cv2.drawContours(original_bgr, contours, -1, (0, 255, 0), 2)  # BGR: verde
    return original_bgr


def encode_png_base64(image_bgr: np.ndarray) -> str:
    success, buffer = cv2.imencode(".png", image_bgr)
    return base64.b64encode(buffer.tobytes()).decode("utf-8")

def build_side_by_side_panel(contour_bgr: np.ndarray, heatmap_bgr: np.ndarray) -> np.ndarray:
    # Aseguramos que ambas tengan la misma altura antes de unirlas
    h = min(contour_bgr.shape[0], heatmap_bgr.shape[0])
    contour_resized = cv2.resize(contour_bgr, (contour_bgr.shape[1], h))
    heatmap_resized = cv2.resize(heatmap_bgr, (heatmap_bgr.shape[1], h))

    # Una franja blanca delgada como separador visual
    separator = np.full((h, 10, 3), 255, dtype=np.uint8)

    return np.hstack([contour_resized, separator, heatmap_resized])

async def validate_and_read_image(file: UploadFile):
    # Comprobacion rapida por content_type declarado
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Tipo de archivo no soportado: {file.content_type}. Usa PNG, JPG o JPEG"
        )

    contents = await file.read()

    # Comprobacion robusta para saber si se puede descodificar como imagen
    try:
        Image.open(io.BytesIO(contents)).verify()
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400,
            detail="El archivo no es una imagen válida o está corrupto.",
        )

    return contents


async def run_interference(contents: bytes, filename: str):
    suffix = Path(filename).suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        dataset = PredictDataset(path=tmp_path)
        predictions = app.state.engine.predict(model=app.state.model, dataset=dataset)
        result = predictions[0]  # una sola pasada por el modelo

        contour_bgr = build_contour_overlay(tmp_path, result.pred_mask)
        heatmap_bgr = build_heatmap_overlay(tmp_path, result.anomaly_map)

        return result, contour_bgr, heatmap_bgr
    finally:
        os.remove(tmp_path)

@app.post(
        "/predict", 
        response_model=PredictionResponse,
        summary="Detecta anomalías en una imagen",
        description=(
            "Recibe una imagen (PNG o JPEG) de un producto y devuelve si el modelo "
            "considera que tiene un defecto, junto con la puntuación de anomalía y "
            "dos visualizaciones (contorno del defecto y mapa de calor) en base64."
    ),
)
async def predict(file: UploadFile = File(...)):
    contents = await validate_and_read_image(file)
    result, contour_bgr, heatmap_bgr = await run_interference(contents, file.filename)
    
    return PredictionResponse(
        filename=file.filename,
        is_anomalous=bool(result.pred_label.item()),
        score=float(result.pred_score.item()),
        contour_image_base64=encode_png_base64(contour_bgr),
        heatmap_image_base64=encode_png_base64(heatmap_bgr),
    )
        
    

@app.post(
        "/predict/preview",
        summary="Vista previa visual del resultado",
        description=(
            "Igual que /predict, pero en vez de JSON devuelve directamente una imagen PNG "
            "con el contorno del defecto y el mapa de calor lado a lado. Pensado para "
            "verificar visualmente desde /docs, no para consumo por otro programa."
        ),
)
async def predict_preview(file: UploadFile = File(...)):
    contents = await validate_and_read_image(file)
    result, contour_bgr, heatmap_bgr = await run_interference(contents, file.filename)

    panel_bgr = build_side_by_side_panel(contour_bgr, heatmap_bgr)
    success, buffer = cv2.imencode(".png", panel_bgr)    

    return Response(
        content=buffer.tobytes(),
        media_type="image/png",
        headers={
            "X-Is-Anomalous": str(bool(result.pred_label.item())),
            "X-Score": str(float(result.pred_score.item())),
        },
    )    
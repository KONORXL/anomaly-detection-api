# app_streamlit.py
import io
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
from anomalib import PrecisionType
from anomalib.models import Patchcore
from anomalib.engine import Engine
from anomalib.data import PredictDataset
from PIL import Image

torch.serialization.add_safe_globals([PrecisionType])

CHECKPOINT_PATH = "./model/bottle_lightning.ckpt"
SAMPLE_DIR = "sample_images"


@st.cache_resource
def load_model():
    model = Patchcore.load_from_checkpoint(CHECKPOINT_PATH, weights_only=True)
    model.eval()
    engine = Engine(enable_progress_bar=False)
    return model, engine


def build_heatmap_overlay(original_image_path: str, anomaly_map) -> np.ndarray:
    amap = anomaly_map.squeeze().cpu().numpy()
    amap_norm = (amap - amap.min()) / (amap.max() - amap.min() + 1e-8)
    amap_uint8 = (amap_norm * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(amap_uint8, cv2.COLORMAP_JET)

    h, w = amap.shape
    original = Image.open(original_image_path).convert("RGB").resize((w, h))
    original_bgr = cv2.cvtColor(np.array(original), cv2.COLOR_RGB2BGR)

    overlay = cv2.addWeighted(original_bgr, 0.6, heatmap_color, 0.4, 0)
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)


def build_contour_overlay(original_image_path: str, pred_mask) -> np.ndarray:
    mask = pred_mask.squeeze().cpu().numpy()
    mask_uint8 = (mask.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = mask.shape
    original = Image.open(original_image_path).convert("RGB").resize((w, h))
    original_bgr = cv2.cvtColor(np.array(original), cv2.COLOR_RGB2BGR)

    cv2.drawContours(original_bgr, contours, -1, (0, 255, 0), 2)
    return cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)


def run_inference(image_bytes: bytes, suffix: str):
    model, engine = load_model()

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(image_bytes)
        tmp_path = tmp.name

    try:
        dataset = PredictDataset(path=tmp_path)
        predictions = engine.predict(model=model, dataset=dataset)
        result = predictions[0]

        contour_rgb = build_contour_overlay(tmp_path, result.pred_mask)
        heatmap_rgb = build_heatmap_overlay(tmp_path, result.anomaly_map)

        return bool(result.pred_label.item()), float(result.pred_score.item()), contour_rgb, heatmap_rgb
    finally:
        os.remove(tmp_path)


st.set_page_config(page_title="Detección de Anomalías", layout="wide")
st.title("🔍 Detección de anomalías en botellas")
st.write("Sube tu propia imagen, o elige una de las imágenes de ejemplo para probar el modelo.")

tab_upload, tab_gallery = st.tabs(["📤 Subir imagen", "🖼️ Usar ejemplo"])

selected_image_bytes = None
selected_suffix = None

with tab_upload:
    uploaded_file = st.file_uploader("Elige una imagen", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        selected_image_bytes = uploaded_file.getvalue()
        selected_suffix = Path(uploaded_file.name).suffix
        st.image(selected_image_bytes, caption="Imagen subida", width=300)

with tab_gallery:
    if os.path.isdir(SAMPLE_DIR):
        sample_files = sorted(os.listdir(SAMPLE_DIR))
        cols = st.columns(5)

        for i, filename in enumerate(sample_files):
            with cols[i % 5]:
                filepath = os.path.join(SAMPLE_DIR, filename)
                st.image(filepath, caption=filename, use_container_width=True)
                if st.button("Elegir", key=f"pick_{filename}"):
                    st.session_state["chosen_sample"] = filename

        chosen_sample = st.session_state.get("chosen_sample")
        if chosen_sample:
            filepath = os.path.join(SAMPLE_DIR, chosen_sample)
            with open(filepath, "rb") as f:
                selected_image_bytes = f.read()
            selected_suffix = Path(chosen_sample).suffix
            st.success(f"Imagen seleccionada: {chosen_sample}")
    else:
        st.info("No hay imágenes de ejemplo disponibles.")

if selected_image_bytes is not None:
    if st.button("Analizar", type="primary"):
        with st.spinner("Analizando imagen..."):
            is_anomalous, score, contour_rgb, heatmap_rgb = run_inference(selected_image_bytes, selected_suffix)

        if is_anomalous:
            st.error(f"⚠️ Anomalía detectada — score: {score:.3f}")
        else:
            st.success(f"✅ Sin anomalías — score: {score:.3f}")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Contorno del defecto")
            st.image(contour_rgb)
        with col2:
            st.subheader("Mapa de calor")
            st.image(heatmap_rgb)
# app_streamlit.py
import base64
import io
import os

import requests
import streamlit as st
from PIL import Image

API_URL = "http://localhost:8000/predict"
SAMPLE_DIR = "sample_images"

st.set_page_config(page_title="Detección de Anomalías", layout="wide")
st.title("🔍 Detección de anomalías en botellas")
st.write("Sube tu propia imagen, o elige una de las imágenes de ejemplo para probar la API.")

tab_upload, tab_gallery = st.tabs(["📤 Subir imagen", "🖼️ Usar ejemplo"])

selected_image_bytes = None
selected_filename = None
selected_content_type = None

with tab_upload:
    uploaded_file = st.file_uploader("Elige una imagen", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        selected_image_bytes = uploaded_file.getvalue()
        selected_filename = uploaded_file.name
        selected_content_type = uploaded_file.type
        st.image(selected_image_bytes, caption="Imagen subida", width=300)

with tab_gallery:
    sample_files = sorted(os.listdir(SAMPLE_DIR))
    cols = st.columns(5)
    chosen_sample = st.session_state.get("chosen_sample")

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
        selected_filename = chosen_sample
        selected_content_type = "image/png"
        st.success(f"Imagen seleccionada: {chosen_sample}")

if selected_image_bytes is not None:
    if st.button("Analizar", type="primary"):
        with st.spinner("Analizando imagen..."):
            files = {"file": (selected_filename, selected_image_bytes, selected_content_type)}
            response = requests.post(API_URL, files=files)

        if response.status_code == 200:
            data = response.json()

            if data["is_anomalous"]:
                st.error(f"⚠️ Anomalía detectada — score: {data['score']:.3f}")
            else:
                st.success(f"✅ Sin anomalías — score: {data['score']:.3f}")

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Contorno del defecto")
                contour_bytes = base64.b64decode(data["contour_image_base64"])
                st.image(Image.open(io.BytesIO(contour_bytes)))

            with col2:
                st.subheader("Mapa de calor")
                heatmap_bytes = base64.b64decode(data["heatmap_image_base64"])
                st.image(Image.open(io.BytesIO(heatmap_bytes)))
        else:
            st.error(f"Error {response.status_code}: {response.json().get('detail', 'Error desconocido')}")
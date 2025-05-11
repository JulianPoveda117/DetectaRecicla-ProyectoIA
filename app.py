import os
os.environ["STREAMLIT_DISABLE_WATCHDOG_WARNINGS"] = "1"
os.environ["STREAMLIT_WATCHED_FILES"] = ""

import streamlit as st
from PIL import Image
import cv2
import numpy as np
from ultralytics import YOLO
import os
import tempfile

# Cargar modelo
model = YOLO("best.pt")
class_names = ["Metales", "Vidrio", "Plástico", "Papel/Cartón", "Residuos peligrosos"]
caneca_colores = ["🟡 Amarilla", "⚪ Blanca", "🔵 Azul", "🩶 Gris", "🔴 Roja"]
img_map = {
    0: ("setUp/metal.png", "setUp/metaltxt.png"),
    1: ("setUp/vidrio.png", "setUp/vidriotxt.png"),
    2: ("setUp/plastico.png", "setUp/plasticotxt.png"),
    3: ("setUp/carton.png", "setUp/cartontxt.png"),
    4: ("setUp/medical.png", "setUp/medicaltxt.png"),
}

# -------------------------- INTERFAZ --------------------------
st.set_page_config(page_title="Clasificador de Residuos", layout="centered")
st.markdown("# ♻️ Detecta y Recicla")
st.markdown("### Aprende a separar correctamente tus residuos 🌱")
st.markdown("Sube una imagen o usa tu cámara para identificar el tipo de residuo y su caneca correspondiente.")

# -------------------------- OPCIÓN 1: SUBIR IMAGEN --------------------------
uploaded_file = st.file_uploader("📁 Sube una imagen del residuo", type=["jpg", "jpeg", "png"])

# -------------------------- OPCIÓN 2: USAR CÁMARA --------------------------
st.markdown("---")
st.markdown("### 📷 O toma una foto desde tu cámara")

camera_image = st.camera_input("Haz clic para tomar una foto")

# -------------------------- DETECCIÓN --------------------------
image_to_process = None

if uploaded_file:
    image_to_process = Image.open(uploaded_file)
elif camera_image:
    image_to_process = Image.open(camera_image)

if image_to_process:
    img_np = np.array(image_to_process)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    st.image(img_np, caption="🖼️ Imagen capturada", use_container_width=True)

    results = model(img_bgr)[0]

    if results.boxes:
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = f"{class_names[cls_id]} ({conf*100:.1f}%)"

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_bgr, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            st.success(f"🔍 Detectado: **{class_names[cls_id]}** con {conf*100:.1f}% de confianza")
            st.info(f"🗑️ Deposítalo en la caneca **{caneca_colores[cls_id]}**")

            col1, col2 = st.columns(2)
            with col1:
                st.image(img_map[cls_id][0], caption="📦 Categoría", use_container_width=True)
            with col2:
                st.image(img_map[cls_id][1], caption="ℹ️ Descripción", use_container_width=True)
    else:
        st.warning("🚫 No se detectó ningún residuo. Intenta con otra imagen.")

    # Mostrar imagen con resultados
    st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="🧠 Resultado del análisis", use_container_width=True)

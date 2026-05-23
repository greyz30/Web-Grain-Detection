import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.title("🌾 Deteksi Jenis Poaceae (Biji-bijian)")
st.write("Unggah foto untuk mendeteksi: Millet, Paddy, Sorgum, Jelai, Oat, Gandum, atau Corn.")

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang diunggah", use_column_width=True)

    st.write("🔍 Sedang mendeteksi...")

    results = model(image, conf=0.3)

    res_plotted = results[0].plot()
    res_plotted = Image.fromarray(res_plotted)

    st.image(res_plotted, caption="Hasil Deteksi", use_column_width=True)

    st.success(f"Terdeteksi {len(results[0].boxes)} objek!")

    # Menampilkan detail deteksi
    if len(results[0].boxes) > 0:
        st.write("📌 Detail Deteksi:")
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id]
            st.write(f"- {label} (Confidence: {conf:.2f})")
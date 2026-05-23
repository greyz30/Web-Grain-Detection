import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np

st.title("🎥 Live Poaceae Detection")
st.write("Tekan tombol di bawah untuk mulai deteksi via Webcam")

# Load model
model = YOLO('best.pt')

# Inisialisasi kamera
run = st.checkbox('Jalankan Kamera')
FRAME_WINDOW = st.image([]) # Tempat untuk menampilkan video

cap = cv2.VideoCapture(0) # 0 adalah ID default webcam laptop

while run:
    ret, frame = cap.read()
    if not ret:
        st.error("Gagal mengakses webcam.")
        break
    frame = cv2.flip(frame, 1)
    # Proses deteksi YOLO pada setiap frame
    # frame adalah BGR (OpenCV), YOLO butuh RGB
    results = model(frame, stream=True)

    for r in results:
        annotated_frame = r.plot() # Menggambar kotak deteksi

    # Tampilkan di Streamlit
    # Kita ubah kembali ke RGB agar warna tidak aneh
    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(annotated_frame)

else:
    st.write("Kamera Berhenti.")
    cap.release()
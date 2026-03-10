# pylint: disable=no-member
import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from fpdf import FPDF
import tempfile

# ===========================
# Page Configuration
# ===========================
st.set_page_config(
    page_title="Plant Disease Detection System",
    page_icon="🌿",
    layout="wide"
)

# ===========================
# Custom CSS
# ===========================
st.markdown("""
<style>
.progress-red > div > div {
    background-color: #ff4b4b;
}
.progress-yellow > div > div {
    background-color: #f7c948;
}
.progress-green > div > div {
    background-color: #2ecc71;
}
</style>
""", unsafe_allow_html=True)

# ===========================
# Load Model
# ===========================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("models/inceptionv3_model.h5")

model = load_model()

# ===========================
# Load Disease Knowledge JSON
# ===========================
with open("data/disease_info.json", "r", encoding="utf-8") as f:
    disease_info = json.load(f)

# ===========================
# Load Class Indices JSON
# ===========================
@st.cache_resource
def load_class_mapping():
    with open("reports/class_indices.json", "r") as f:
        class_indices = json.load(f)
    return {v: k for k, v in class_indices.items()}

index_to_class = load_class_mapping()

# ===========================
# Helper Functions
# ===========================
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

def confidence_color(conf):
    if conf < 40:
        return "progress-red"
    if conf < 70:
        return "progress-yellow"
    return "progress-green"

def normalize_class_name(name: str) -> str:
    """Normalize predicted class name for safe JSON lookup"""
    return name.strip().replace("__", "_").rstrip("_")

def generate_pdf(image, disease, confidence, info):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Plant Disease Detection Report", ln=True)

    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Disease: {disease.replace('___', ' - ')}", ln=True)
    pdf.cell(0, 10, f"Confidence: {confidence:.2f}%", ln=True)
    pdf.ln(5)

    pdf.multi_cell(0, 8, f"Description:\n{info['description']}")
    pdf.ln(3)

    pdf.multi_cell(0, 8, "Symptoms:\n- " + "\n- ".join(info["symptoms"]))
    pdf.ln(3)

    pdf.multi_cell(0, 8, "Remedies:\n- " + "\n- ".join(info["remedies"]))
    pdf.ln(5)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_img:
        image.save(tmp_img.name)
        pdf.image(tmp_img.name, x=10, y=pdf.get_y(), w=80)

    pdf_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name
    pdf.output(pdf_path)
    return pdf_path

# ===========================
# UI Layout
# ===========================
st.markdown("# 🌿 Plant Disease Detection System")
st.markdown(
    "Upload a **single leaf image** to detect plant disease using "
    "**Deep Learning (InceptionV3)**."
)
st.info(
    "📌 Upload a clear leaf image with good lighting and plain background "
    "for best results."
)

uploaded_file = st.file_uploader(
    "📤 Upload Leaf Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    col1, col2 = st.columns([1, 1])

    image = Image.open(uploaded_file).convert("RGB")

    with col1:
        st.image(image, caption="Uploaded Leaf Image", use_container_width=True)

    with col2:
        with st.spinner("🔍 Analyzing leaf image..."):
            processed_image = preprocess_image(image)
            predictions = model.predict(processed_image)[0]
            predicted_index = int(np.argmax(predictions))
            confidence = predictions[predicted_index] * 100

            raw_disease = index_to_class[predicted_index]
            normalized_disease = normalize_class_name(raw_disease)

            info = disease_info.get(normalized_disease)

            # Fallback knowledge (never break UI)
            if not info:
                info = {
                    "description": (
                        "This disease affects plant health and may reduce crop yield. "
                        "The prediction is based on learned visual patterns."
                    ),
                    "symptoms": [
                        "Visible discoloration on leaves",
                        "Spots or texture changes"
                    ],
                    "remedies": [
                        "Consult an agricultural expert",
                        "Use appropriate fungicide or treatment"
                    ]
                }

        st.markdown("## 🔎 Prediction Result")
        st.success(f"**Disease:** {raw_disease.replace('___', ' - ')}")

        st.markdown("### 📊 Prediction Confidence")
        progress_class = confidence_color(confidence)
        st.markdown(f'<div class="{progress_class}">', unsafe_allow_html=True)
        st.progress(int(confidence))
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown(f"### **{confidence:.2f}%**")

        st.markdown("---")

        st.markdown("### 📘 Disease Description")
        st.write(info["description"])

        st.markdown("### ⚠️ Symptoms")
        for s in info["symptoms"]:
            st.write(f"- {s}")

        st.markdown("### 🩺 Remedies")
        for r in info["remedies"]:
            st.write(f"- {r}")

        pdf_path = generate_pdf(image, raw_disease, confidence, info)
        with open(pdf_path, "rb") as f:
            st.download_button(
                "📄 Download Detailed Report (PDF)",
                f,
                file_name="plant_disease_report.pdf",
                mime="application/pdf"
            )

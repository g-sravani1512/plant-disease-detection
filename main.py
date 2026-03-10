# ===========================
# Plant Disease Detection Backend
# Framework: Flask
# ===========================

from flask import Flask, render_template, request, jsonify, send_file
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import io
import base64
import os
import gdown

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER

# ===========================
# Flask App Configuration
# ===========================

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# ===========================
# Model Configuration
# ===========================

MODEL_PATH = "models/inceptionv3_model.h5"
MODEL_URL = "https://drive.google.com/uc?id=1JmJbkLF4WQLkgSKJmvy2ac6CfELxarb4"

model = None

def ensure_model_exists():
    if not os.path.exists(MODEL_PATH):
        print("Model not found. Downloading model...")
        os.makedirs("models", exist_ok=True)
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
        print("✓ Model downloaded successfully")

# ===========================
# Load Model
# ===========================

def load_model():
    global model
    ensure_model_exists()
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✓ Model loaded successfully")

load_model()

# ===========================
# Load Disease Info
# ===========================

with open('data/disease_info.json', 'r', encoding='utf-8') as f:
    disease_info = json.load(f)

# ===========================
# Load Class Mapping
# ===========================

def load_class_mapping():
    with open('reports/class_indices.json', 'r') as f:
        data = json.load(f)

    if isinstance(list(data.values())[0], int):
        return {v: k for k, v in data.items()}

    return {int(k): v for k, v in data.items()}

index_to_class = load_class_mapping()

# ===========================
# Helper Functions
# ===========================

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image):
    image = image.convert('RGB')
    image = image.resize((224, 224))
    img = np.array(image) / 255.0
    return np.expand_dims(img, axis=0)

def canonicalize_name(name):
    return (
        name.lower()
        .replace('(', '')
        .replace(')', '')
        .replace('-', '*')
        .replace(' ', '*')
        .replace('**', '*')
        .replace('***', '*')
        .strip('*')
    )

def fallback_info():
    return {
        "description": "This disease affects plant health and may reduce crop yield. The prediction is based on learned visual patterns from similar cases.",
        "symptoms": [
            "Visible discoloration on leaves",
            "Spots or texture changes on leaf surface",
            "Possible wilting or deformation"
        ],
        "remedies": [
            "Consult an agricultural expert for proper diagnosis",
            "Remove affected leaves to prevent spread",
            "Use appropriate fungicide or treatment as recommended",
            "Ensure proper plant nutrition and watering"
        ]
    }

def get_disease_info(raw_name):
    if not raw_name or raw_name == "Unknown":
        return fallback_info()

    pred_key = canonicalize_name(raw_name)

    for key, value in disease_info.items():
        if canonicalize_name(key) == pred_key:
            return value

    return fallback_info()

# ===========================
# PDF Generation
# ===========================

def generate_pdf_report(image_data, disease, confidence, info):
    buffer = io.BytesIO()

    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'Title',
        parent=styles['Heading1'],
        alignment=TA_CENTER,
        textColor=colors.HexColor("#2c5f2d")
    )

    story = []

    story.append(Paragraph("Plant Disease Detection Report", title_style))
    story.append(Spacer(1, 0.3 * inch))

    table = Table(
        [
            ["Disease", disease],
            ["Confidence", f"{confidence:.2f}%"]
        ],
        colWidths=[2.5 * inch, 4 * inch]
    )

    table.setStyle(
        TableStyle([
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor("#e8f5e9")),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold')
        ])
    )

    story.append(table)
    story.append(Spacer(1, 0.3 * inch))

    image_bytes = base64.b64decode(image_data.split(',')[1])

    story.append(RLImage(io.BytesIO(image_bytes), width=3 * inch, height=3 * inch))
    story.append(Spacer(1, 0.3 * inch))

    story.append(Paragraph("Description", styles['Heading2']))
    story.append(Paragraph(info['description'], styles['Normal']))

    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("Symptoms", styles['Heading2']))
    for s in info['symptoms']:
        story.append(Paragraph(f"• {s}", styles['Normal']))

    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("Remedies", styles['Heading2']))
    for r in info['remedies']:
        story.append(Paragraph(f"• {r}", styles['Normal']))

    doc.build(story)
    buffer.seek(0)

    return buffer

# ===========================
# Routes
# ===========================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/working')
def working():
    return render_template('working.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' in request.files:
            file = request.files['image']

            if not allowed_file(file.filename):
                return jsonify({'success': False, 'error': 'Invalid file'})

            image = Image.open(file.stream)

        elif 'image_data' in request.form:
            image_bytes = base64.b64decode(request.form['image_data'].split(',')[1])
            image = Image.open(io.BytesIO(image_bytes))

        else:
            return jsonify({'success': False, 'error': 'No image provided'})

        processed = preprocess_image(image)

        preds = model.predict(processed)[0]

        predicted_index = int(np.argmax(preds))
        confidence = float(preds[predicted_index] * 100)

        raw_disease = index_to_class.get(predicted_index, "Unknown")
        info = get_disease_info(raw_disease)

        display_name = raw_disease.replace('___', ' - ').replace('_', ' ')

        return jsonify({
            'success': True,
            'disease': display_name,
            'confidence': round(confidence, 2),
            'description': info['description'],
            'symptoms': info['symptoms'],
            'remedies': info['remedies']
        })

    except Exception as e:
        print("Prediction error:", e)
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/generate-pdf', methods=['POST'])
def generate_pdf():
    data = request.get_json()

    pdf = generate_pdf_report(
        data['image'],
        data['disease'],
        float(data['confidence']),
        {
            'description': data['description'],
            'symptoms': data['symptoms'],
            'remedies': data['remedies']
        }
    )

    return send_file(
        pdf,
        mimetype='application/pdf',
        as_attachment=True,
        download_name='plant_disease_report.pdf'
    )

# ===========================
# Run Server
# ===========================

if __name__ == "__main__":
    print("🌿 Plant Disease Detection running at http://127.0.0.1:5000")
    app.run(debug=True)
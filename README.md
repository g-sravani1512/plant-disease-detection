# 🌿 Plant Disease Detection System

An AI-powered **Plant Disease Detection Web Application** built using **Flask, TensorFlow, and Deep Learning**.
The system analyzes plant leaf images and predicts the disease using trained **Convolutional Neural Network (CNN)** and **InceptionV3** models.

This project helps farmers, researchers, and agriculture enthusiasts **detect plant diseases early** and take preventive actions to protect crops.

---

# 📌 Features

* 🌱 Upload plant leaf images for disease detection
* 🤖 Deep Learning based classification
* 🧠 Supports **CNN** and **InceptionV3 Transfer Learning models**
* 📊 Displays disease description, symptoms, and remedies
* 📄 Generate downloadable **PDF disease report**
* ☁️ Automatic **model download from Google Drive**
* 🌐 Simple and responsive web interface

---

# 🧠 Models Used

Two deep learning models were trained for this project.

### 1️⃣ CNN Model

A custom Convolutional Neural Network trained on the PlantVillage dataset.

**Download CNN Model:**

```
https://drive.google.com/file/d/1TEZz6dUAgi3ZERD4MNt-_X05OXKgmBx1/view?usp=sharing
```

---

### 2️⃣ InceptionV3 Model

Transfer learning model based on the **InceptionV3 architecture** pretrained on ImageNet.

**Download InceptionV3 Model:**

```
https://drive.google.com/file/d/1JmJbkLF4WQLkgSKJmvy2ac6CfELxarb4/view?usp=drive_link
```

---

# 📊 Dataset

This project uses the **PlantVillage dataset**, which contains thousands of labeled plant leaf images across multiple disease classes.

**Dataset Source (Kaggle):**

```
https://www.kaggle.com/datasets/emmarex/plantdisease
```

The dataset includes:

* Healthy plant leaves
* Diseased plant leaves
* Multiple crop species
* Multiple disease categories

---

# 🗂 Project Folder Structure

```
plant-disease-detection
│
├── data
│   └── disease_info.json
│
├── reports
│   └── class_indices.json
│
├── src
│   └── training scripts
│
├── static
│   ├── css
│   ├── js
│   └── images
│
├── templates
│   ├── index.html
│   └── working.html
│
├── streamlit_app
│   └── alternative UI
│
├── uploads
│   └── uploaded images
│
├── main.py
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation

### 1️⃣ Clone the repository

```
git clone https://github.com/g-sravani1512/plant-disease-detection.git
```

```
cd plant-disease-detection
```

---

### 2️⃣ Create virtual environment

```
python -m venv venv
```

Activate environment

Windows:

```
venv\Scripts\activate
```

Mac/Linux:

```
source venv/bin/activate
```

---

### 3️⃣ Install dependencies

```
pip install -r requirements.txt
```

---

# ▶️ Running the Application

Start the Flask server:

```
python main.py
```

The application will start at:

```
http://127.0.0.1:5000
```

---

# 🔄 Automatic Model Download

When the server starts, the application checks if the model exists locally.

If the model is missing:

1️⃣ The model is downloaded from **Google Drive**
2️⃣ Saved into the `models/` directory
3️⃣ Loaded automatically by TensorFlow

This keeps the GitHub repository **lightweight** and avoids GitHub's file size limitations.

---

# 📄 PDF Report Generation

After prediction, the system generates a **disease analysis report** including:

* Disease Name
* Prediction Confidence
* Disease Description
* Symptoms
* Recommended Remedies

Users can download the report as a **PDF document**.

---

# 🧰 Technologies Used

* Python
* Flask
* TensorFlow / Keras
* NumPy
* Pillow
* ReportLab
* HTML / CSS / JavaScript
* Bootstrap

---

# 🚀 Future Improvements

* Mobile application integration
* Real-time camera disease detection
* IoT-based plant monitoring system
* Multi-language farmer support
* More crop and disease classes

---

# 👩‍💻 Author

**Sravani G**

GitHub:
https://github.com/g-sravani1512

---

# 📜 License

This project is developed for **educational and research purposes**.

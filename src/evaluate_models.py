"""
Evaluation & Comparison of Models
CNN vs InceptionV3
"""

import tensorflow as tf
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ---------------------------
# Paths & Settings
# ---------------------------
VAL_DIR = "dataset/PlantVillage/val"
IMAGE_SIZE = 224
BATCH_SIZE = 32

# ---------------------------
# Validation Data Generator
# ---------------------------
val_datagen = ImageDataGenerator(rescale=1.0 / 255)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False,
)

# ---------------------------
# Load Models
# ---------------------------
cnn_model = tf.keras.models.load_model("models/cnn_model.h5")
inception_model = tf.keras.models.load_model("models/inceptionv3_model.h5")

# ---------------------------
# Evaluate CNN
# ---------------------------
print("\nEvaluating CNN model...")
cnn_loss, cnn_acc = cnn_model.evaluate(val_generator, verbose=1)

# ---------------------------
# Evaluate InceptionV3
# ---------------------------
print("\nEvaluating InceptionV3 model...")
inc_loss, inc_acc = inception_model.evaluate(val_generator, verbose=1)

# ---------------------------
# Results Table
# ---------------------------
results = pd.DataFrame({
    "Model": ["CNN (From Scratch)", "InceptionV3 (Transfer Learning)"],
    "Validation Accuracy (%)": [cnn_acc * 100, inc_acc * 100],
    "Validation Loss": [cnn_loss, inc_loss],
})

print("\n--- Model Comparison ---")
print(results)

# ---------------------------
# Save Results
# ---------------------------
results.to_csv("reports/model_comparison.csv", index=False)

print("\nEvaluation results saved to reports/model_comparison.csv")

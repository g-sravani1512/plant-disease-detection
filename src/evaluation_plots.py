"""
Confusion Matrix & Classification Report
Final Evaluation of InceptionV3 Model
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report

# ---------------------------
# Paths & Settings
# ---------------------------
VAL_DIR = "dataset/PlantVillage/val"
IMAGE_SIZE = 224
BATCH_SIZE = 32

# ---------------------------
# Validation Generator
# ---------------------------
val_datagen = ImageDataGenerator(rescale=1.0 / 255)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False,
)

class_names = list(val_generator.class_indices.keys())

# ---------------------------
# Load Final Model
# ---------------------------
model = tf.keras.models.load_model("models/inceptionv3_model.h5")

# ---------------------------
# Predictions
# ---------------------------
y_true = val_generator.classes
pred_probs = model.predict(val_generator)
y_pred = np.argmax(pred_probs, axis=1)

# ---------------------------
# Confusion Matrix
# ---------------------------
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(14, 12))
sns.heatmap(
    cm,
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
    cbar=True,
)
plt.title("Confusion Matrix - InceptionV3")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()

# Save confusion matrix
plt.savefig("reports/confusion_matrix_inceptionv3.png", dpi=300)
plt.show()

print("Confusion matrix saved to reports/confusion_matrix_inceptionv3.png")

# ---------------------------
# Classification Report
# ---------------------------
report = classification_report(
    y_true, y_pred, target_names=class_names
)

print("\nClassification Report:\n")
print(report)

# Save classification report
with open("reports/classification_report.txt", "w") as f:
    f.write(report)

print("Classification report saved to reports/classification_report.txt")

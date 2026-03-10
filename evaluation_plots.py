import matplotlib
matplotlib.use("Agg")

# ======================================
# IMPORTS
# ======================================
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.preprocessing import label_binarize
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# ======================================
# PATHS
# ======================================

DATASET_PATH = "dataset/PlantVillage"
TRAIN_PATH = os.path.join(DATASET_PATH, "train")
VAL_PATH = os.path.join(DATASET_PATH, "val")

CNN_MODEL_PATH = "models/cnn_model.h5"
INCEPTION_MODEL_PATH = "models/inceptionv3_model.h5"

REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# ======================================
# COUNT IMAGES
# ======================================

def count_images(folder):
    total = 0
    for root, dirs, files in os.walk(folder):
        total += len([f for f in files if f.lower().endswith((".jpg", ".png", ".jpeg"))])
    return total

total_images = count_images(DATASET_PATH)
train_images = count_images(TRAIN_PATH)
val_images = count_images(VAL_PATH)

print("Total Images:", total_images)
print("Training Images:", train_images)
print("Validation Images:", val_images)

with open(os.path.join(REPORTS_DIR, "final_summary.txt"), "w") as f:
    f.write(f"Total Images: {total_images}\n")
    f.write(f"Training Images: {train_images}\n")
    f.write(f"Validation Images: {val_images}\n")
    f.write(f"Train Ratio: {round(train_images/total_images,2)}\n")
    f.write(f"Validation Ratio: {round(val_images/total_images,2)}\n")

# ======================================
# DATA GENERATOR
# ======================================

datagen = ImageDataGenerator(rescale=1./255)

val_generator = datagen.flow_from_directory(
    VAL_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

num_classes = val_generator.num_classes
class_names = list(val_generator.class_indices.keys())

# ======================================
# LOAD MODELS (FIXED WARNING)
# ======================================

cnn_model = load_model(CNN_MODEL_PATH, compile=False)
inception_model = load_model(INCEPTION_MODEL_PATH, compile=False)

# Compile manually to remove warnings
cnn_model.compile(
    optimizer=Adam(),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

inception_model.compile(
    optimizer=Adam(),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ======================================
# EVALUATE MODELS
# ======================================

cnn_loss, cnn_acc = cnn_model.evaluate(val_generator, verbose=1)
inception_loss, inception_acc = inception_model.evaluate(val_generator, verbose=1)

print("CNN Accuracy:", cnn_acc)
print("InceptionV3 Accuracy:", inception_acc)

# ======================================
# PREDICTIONS
# ======================================

val_generator.reset()
cnn_pred_probs = cnn_model.predict(val_generator, verbose=1)
cnn_preds = np.argmax(cnn_pred_probs, axis=1)

val_generator.reset()
inception_pred_probs = inception_model.predict(val_generator, verbose=1)
inception_preds = np.argmax(inception_pred_probs, axis=1)

y_true = val_generator.classes

# ======================================
# CORRECT PREDICTION COUNT
# ======================================

cnn_correct = np.sum(cnn_preds == y_true)
inception_correct = np.sum(inception_preds == y_true)

print("CNN Correct Predictions:", cnn_correct)
print("Inception Correct Predictions:", inception_correct)

with open(os.path.join(REPORTS_DIR, "evaluation_summary.txt"), "w") as f:
    f.write(f"CNN Accuracy: {cnn_acc}\n")
    f.write(f"InceptionV3 Accuracy: {inception_acc}\n")
    f.write(f"CNN Correct Predictions: {cnn_correct}\n")
    f.write(f"Inception Correct Predictions: {inception_correct}\n")

# ======================================
# ACCURACY COMPARISON GRAPH
# ======================================

plt.figure()
plt.bar(["CNN", "InceptionV3"], [cnn_acc, inception_acc])
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.savefig(os.path.join(REPORTS_DIR, "model_comparison.png"))
plt.close()

# ======================================
# CONFUSION MATRIX - CNN
# ======================================

cm_cnn = confusion_matrix(y_true, cnn_preds)

plt.figure(figsize=(18, 15))
sns.heatmap(
    cm_cnn,
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.title("Confusion Matrix - CNN")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, "confusion_matrix_cnn.png"))
plt.close()

print("CNN Confusion Matrix saved successfully.")

# ======================================
# ROC CURVE (Micro Average - Inception)
# ======================================

y_true_bin = label_binarize(y_true, classes=range(num_classes))

fpr, tpr, _ = roc_curve(
    y_true_bin.ravel(),
    inception_pred_probs.ravel()
)

roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, linewidth=2, label=f"AUC = {roc_auc:.4f}")
plt.plot([0,1], [0,1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - InceptionV3")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig(os.path.join(REPORTS_DIR, "roc_curve.png"))
plt.close()

print("ROC Curve saved successfully.")
print("\nAll Evaluation Reports Saved in 'reports/' Folder ✅")

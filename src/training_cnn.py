# pylint: disable=no-member
"""
Train CNN model for Plant Disease Detection
"""

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from building_cnn import build_cnn_model

# Paths
TRAIN_DIR = "dataset/PlantVillage/train"
VAL_DIR = "dataset/PlantVillage/val"

IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 5  # keep small for first training

# ---------------------------
# Data Generators
# ---------------------------
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
)

val_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
)

# ---------------------------
# Build & Compile Model
# ---------------------------
model = build_cnn_model()

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# ---------------------------
# Train Model
# ---------------------------
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=3,
    steps_per_epoch=200,
    validation_steps=50,
)

# ---------------------------
# Save Model
# ---------------------------
model.save("models/cnn_model.h5")

print("CNN model training completed and saved successfully.")

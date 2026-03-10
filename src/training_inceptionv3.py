# pylint: disable=no-member
"""
InceptionV3 Transfer Learning for Plant Disease Detection
"""

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ---------------------------
# Paths
# ---------------------------
TRAIN_DIR = "dataset/PlantVillage/train"
VAL_DIR = "dataset/PlantVillage/val"

IMAGE_SIZE = 224
BATCH_SIZE = 16   # smaller batch for CPU safety
EPOCHS = 8
NUM_CLASSES = 38

# ---------------------------
# Data Generators
# ---------------------------
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=30,
    zoom_range=0.3,
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
# Load InceptionV3 Base Model
# ---------------------------
base_model = tf.keras.applications.InceptionV3(
    weights="imagenet",
    include_top=False,
    input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
)

# Freeze base model (IMPORTANT)
base_model.trainable = False

# ---------------------------
# Build Full Model
# ---------------------------
inputs = tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
x = base_model(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(256, activation="relu")(x)
x = tf.keras.layers.Dropout(0.5)(x)
outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = tf.keras.Model(inputs, outputs)

# ---------------------------
# Compile Model
# ---------------------------
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
    epochs=EPOCHS,
    steps_per_epoch=400,
    validation_steps=100,
)

# ---------------------------
# Save Model
# ---------------------------
model.save("models/inceptionv3_model.h5")

print("InceptionV3 model trained and saved successfully.")

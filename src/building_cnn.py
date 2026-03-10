"""
CNN model definition
"""

import tensorflow as tf

IMAGE_SIZE = 224
NUM_CLASSES = 38


def build_cnn_model():
    model = tf.keras.models.Sequential()

    model.add(
        tf.keras.layers.Conv2D(
            32,
            (3, 3),
            activation="relu",
            input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        )
    )
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))

    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))

    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation="relu"))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(NUM_CLASSES, activation="softmax"))

    return model


if __name__ == "__main__":
    cnn_model = build_cnn_model()
    cnn_model.summary()

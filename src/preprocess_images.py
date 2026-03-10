import os
import numpy as np
from PIL import Image

# Paths
DATASET_PATH = "dataset/PlantVillage/train"
IMAGE_SIZE = 224

images = []
labels = []

class_names = os.listdir(DATASET_PATH)
print("Total classes:", len(class_names))

for label, class_name in enumerate(class_names):
    class_path = os.path.join(DATASET_PATH, class_name)

    # Take only first 5 images per class (for now)
    image_files = os.listdir(class_path)[:5]

    for image_file in image_files:
        image_path = os.path.join(class_path, image_file)

        # Load image and convert to RGB
        img = Image.open(image_path).convert("RGB")

        # Resize image
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE))

        # Convert to numpy array
        img_array = np.array(img, dtype=np.float32)

        # Normalize pixel values
        img_array = img_array / 255.0

        images.append(img_array)
        labels.append(label)

images = np.array(images)
labels = np.array(labels)

print("Images shape:", images.shape)
print("Labels shape:", labels.shape)

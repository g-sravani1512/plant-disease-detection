import os
from PIL import Image
import matplotlib.pyplot as plt

DATASET_PATH = "dataset/PlantVillage"

# Go inside train folder
train_path = os.path.join(DATASET_PATH, "train")

# List disease classes
classes = os.listdir(train_path)
print("Classes found:", classes[:5])

# Pick one class
sample_class = classes[0]

# Go inside that class folder
class_path = os.path.join(train_path, sample_class)

# Pick one image file
image_name = os.listdir(class_path)[0]
image_path = os.path.join(class_path, image_name)

# Load and show image
img = Image.open(image_path)
plt.imshow(img)
plt.axis("off")
plt.title(sample_class)
plt.show()

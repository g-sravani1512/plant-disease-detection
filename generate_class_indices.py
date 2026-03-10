import json
from tensorflow.keras.preprocessing.image import ImageDataGenerator

gen = ImageDataGenerator().flow_from_directory(
    "dataset/PlantVillage/train",
    target_size=(224, 224),
    batch_size=1
)

with open("reports/class_indices.json", "w") as f:
    json.dump(gen.class_indices, f, indent=4)

print("✅ class_indices.json generated successfully")

import os
import random
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# Base directory
base_dir = os.path.dirname(__file__)

# Load metadata
df = pd.read_csv(os.path.join(base_dir, "data", "HAM10000_metadata.csv"))

# Image folders
folder1 = os.path.join(base_dir, "data", "HAM10000_images_part_1")
folder2 = os.path.join(base_dir, "data", "HAM10000_images_part_2")

# Pick 6 random images
samples = df.groupby("dx").sample(n=1, random_state=42)

plt.figure(figsize=(14,10))

for i, (_, row) in enumerate(samples.iterrows()):
    image_name = row["image_id"] + ".jpg"

    image_path = os.path.join(folder1, image_name)

    if not os.path.exists(image_path):
        image_path = os.path.join(folder2, image_name)

    image = Image.open(image_path)

    plt.subplot(3, 3, i + 1)
    plt.imshow(image)
    plt.title(row["dx"])
    plt.axis("off")

plt.tight_layout()
plt.show()
from dataset import HAM10000Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Create dataset
dataset = HAM10000Dataset()

print("Dataset size:", len(dataset))

image, label = dataset[0]

print("Image shape:", image.shape)
print("Label:", label)

# Create DataLoader
loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True
)

images, labels = next(iter(loader))

print("\nBatch Shapes")
print("Images:", images.shape)
print("Labels:", labels.shape)

image, label = dataset[0]

print("Image shape:", image.shape)
print("Numeric label:", label)
print("Diagnosis:", dataset.df.iloc[0]["dx"])

print(dataset.label_map)
print(image.min())
print(image.max())

image, label = dataset[0]

plt.imshow(image.permute(1, 2, 0))
plt.title(f"Label: {label}")
plt.axis("off")
plt.show()
import os
import pandas as pd

# Get the folder where this script is located
base_dir = os.path.dirname(__file__)

# Build the full path to the CSV file
csv_path = os.path.join(base_dir, "data", "HAM10000_metadata.csv")

df = pd.read_csv(csv_path)

print("=" * 50)
print("HAM10000 DATASET OVERVIEW")
print("=" * 50)

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nDiagnosis Distribution:")
print(df["dx"].value_counts())

print("\nMissing Values:")
print(df.isnull().sum())
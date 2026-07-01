import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('ham10000/data/HAM10000_metadata.csv')

print("=== DUPLICATE ANALYSIS ===")
print(f"Total images: {len(df)}")
print(f"Unique lesion_id: {df['lesion_id'].nunique()}")
print(f"Images per lesion (value counts):")
print(df.groupby('lesion_id').size().value_counts().head(10))

# How many lesions have >1 image?
multi = df.groupby('lesion_id').size()
print(f"\nLesions with >1 image: {(multi > 1).sum()}")
print(f"Lesions with exactly 1 image: {(multi == 1).sum()}")

# Get unique lesions with their diagnosis
lesion_df = df.groupby('lesion_id').agg(
    dx=('dx', 'first')
).reset_index()

print(f"\nUnique lesions: {len(lesion_df)}")
print(lesion_df['dx'].value_counts())

# Step 1: split unique lesions into train (80%) and temp (20%)
train_lesions, temp_lesions = train_test_split(
    lesion_df['lesion_id'],
    test_size=0.20,
    stratify=lesion_df['dx'],
    random_state=42
)

# Step 2: split temp evenly into val (10%) and test (10%)
val_lesions, test_lesions = train_test_split(
    temp_lesions,
    test_size=0.50,
    stratify=lesion_df.loc[
        lesion_df['lesion_id'].isin(temp_lesions), 'dx'
    ],
    random_state=42
)

# Map back to all images
df['split'] = 'train'
df.loc[df['lesion_id'].isin(val_lesions),  'split'] = 'val'
df.loc[df['lesion_id'].isin(test_lesions), 'split'] = 'test'

# Verify — print counts per split
print("\n=== SPLIT COUNTS ===")
print(df['split'].value_counts())
print("\n=== CLASS DISTRIBUTION PER SPLIT ===")
print(df.groupby(['split', 'dx']).size().unstack(fill_value=0))

# Save — NEVER regenerate this, always load it
df.to_csv('ham10000/data/HAM10000_split.csv', index=False)
print("\nSaved to ham10000/data/HAM10000_split.csv")
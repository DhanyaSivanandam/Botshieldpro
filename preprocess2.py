import pandas as pd
import numpy as np

# =========================
# 1. Load Dataset
# =========================
df = pd.read_csv("preprocessed_dataset.csv")   # change filename if needed

print("Original Shape:", df.shape)

# =========================
# 2. Drop Unnecessary Columns
# =========================
columns_to_drop = ["test_set_1", "test_set_2"]  # remove if exist

df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

# =========================
# 3. Separate Label
# =========================
label_column = "label"

if label_column not in df.columns:
    raise Exception("Label column not found!")

# =========================
# 4. Handle Missing Values Properly
# =========================

# Separate numeric and categorical
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

# Remove label from numeric columns if present
if label_column in num_cols:
    num_cols.remove(label_column)

# Fill numeric columns with median
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

# Fill categorical columns with mode
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# =========================
# 5. Convert Boolean to 0/1
# =========================
df = df.replace({True: 1, False: 0})

# =========================
# 6. Ensure Label is Integer
# =========================
df[label_column] = df[label_column].astype(int)

# =========================
# 7. Final Check
# =========================
print("After Cleaning Shape:", df.shape)
print("\nLabel Distribution:")
print(df["label"].value_counts())

# =========================
# 8. Save Clean Dataset
# =========================
df.to_csv("clean_dataset.csv", index=False)

print("\nClean dataset saved successfully.")

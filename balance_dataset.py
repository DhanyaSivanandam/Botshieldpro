import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# =========================
# 1. Load Clean Dataset
# =========================
df = pd.read_csv("clean_dataset.csv")

print("Loaded Shape:", df.shape)

# =========================
# 2. Split Features & Label
# =========================
X = df.drop("label", axis=1)
y = df["label"]

# =========================
# 3. Train Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nBefore SMOTE:")
print(y_train.value_counts())

# =========================
# 4. Apply SMOTE (Only on Train)
# =========================
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

print("\nAfter SMOTE:")
print(y_train_bal.value_counts())

# =========================
# 5. Save Balanced Training Data
# =========================
train_balanced = pd.concat([X_train_bal, y_train_bal], axis=1)
train_balanced.to_csv("balanced_train_dataset.csv", index=False)

print("\nBalanced dataset saved successfully.")

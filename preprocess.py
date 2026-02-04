import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("final_dataset.csv")

print("Original Shape:", df.shape)

# ----------------------------
# 1️⃣ Drop unnecessary columns
# ----------------------------

columns_to_drop = [
    "id", "name", "screen_name", "description",
    "url", "profile_image_url", "profile_banner_url"
]

for col in columns_to_drop:
    if col in df.columns:
        df.drop(col, axis=1, inplace=True)

# ----------------------------
# 2️⃣ Handle missing values properly
# ----------------------------

# Fill numeric columns with 0
num_cols = df.select_dtypes(include=[np.number]).columns
df[num_cols] = df[num_cols].fillna(0)

# Fill string columns with empty string
str_cols = df.select_dtypes(include=["object", "string"]).columns
df[str_cols] = df[str_cols].fillna("")

# ----------------------------
# 3️⃣ Convert created_at to account_age
# ----------------------------
# ----------------------------
# 3️⃣ Convert created_at to account_age
# ----------------------------

if "created_at" in df.columns:
    
    # Convert to datetime and remove timezone
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
    df["created_at"] = df["created_at"].dt.tz_localize(None)

    # Use timezone-naive current time
    current_time = pd.Timestamp.now()

    df["account_age_days"] = (current_time - df["created_at"]).dt.days
    df["account_age_days"] = df["account_age_days"].fillna(0)

    df.drop("created_at", axis=1, inplace=True)


# ----------------------------
# 4️⃣ Feature Engineering
# ----------------------------

if "followers_count" in df.columns and "friends_count" in df.columns:
    df["follower_friend_ratio"] = df["followers_count"] / (df["friends_count"] + 1)

if "statuses_count" in df.columns and "followers_count" in df.columns:
    df["tweets_per_follower"] = df["statuses_count"] / (df["followers_count"] + 1)

# ----------------------------
# 5️⃣ Convert boolean columns
# ----------------------------

bool_cols = df.select_dtypes(include=["bool"]).columns
for col in bool_cols:
    df[col] = df[col].astype(int)

# ----------------------------
# 6️⃣ Keep only numeric columns for ML
# ----------------------------

df = df.select_dtypes(include=[np.number])
# ----------------------------
# Save cleaned dataset BEFORE scaling
# ----------------------------

df.to_csv("preprocessed_dataset.csv", index=False)
print("Preprocessed dataset saved successfully.")


print("After Cleaning Shape:", df.shape)

# ----------------------------
# 7️⃣ Separate features and label
# ----------------------------

X = df.drop("label", axis=1)
y = df["label"]
df = df.drop(["test_set_1", "test_set_2"], axis=1)


# ----------------------------
# 8️⃣ Train Test Split
# ----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ----------------------------
# 9️⃣ Feature Scaling
# ----------------------------

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Preprocessing Completed Successfully")
print("Train Shape:", X_train.shape)
print("Test Shape:", X_test.shape)

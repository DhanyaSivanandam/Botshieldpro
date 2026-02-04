import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

print("Starting Preprocessing...")

# ==============================
# 1️⃣ Load Dataset
# ==============================
df = pd.read_csv("final_dataset.csv")

print("Initial Shape:", df.shape)


# ==============================
# 2️⃣ Handle Missing Values
# ==============================

# Fill numeric columns with 0
numeric_cols = df.select_dtypes(include=np.number).columns
df[numeric_cols] = df[numeric_cols].fillna(0)

# Fill object and string columns with empty string
text_cols = df.select_dtypes(include=["object", "string"]).columns
df[text_cols] = df[text_cols].fillna("")


# ==============================
# 3️⃣ Feature Engineering
# ==============================

# Followers-Friends Ratio
if "followers_count" in df.columns and "friends_count" in df.columns:
    df["followers_friends_ratio"] = df["followers_count"] / (df["friends_count"] + 1)
else:
    df["followers_friends_ratio"] = 0


# ==============================
# Account Age (100% FIXED)
# ==============================

if "created_at" in df.columns:
    # Convert to datetime and force everything to UTC
    df["created_at"] = pd.to_datetime(
        df["created_at"],
        errors="coerce",
        utc=True
    )

    # Remove timezone info completely (make tz-naive)
    df["created_at"] = df["created_at"].dt.tz_convert(None)

    # Now safe subtraction
    current_time = pd.Timestamp.now()

    df["account_age_days"] = (
        current_time - df["created_at"]
    ).dt.days

    df["account_age_days"] = df["account_age_days"].fillna(0)

else:
    df["account_age_days"] = 0



# Activity Rate
if "statuses_count" in df.columns:
    df["activity_rate"] = df["statuses_count"] / (df["account_age_days"] + 1)
else:
    df["activity_rate"] = 0


# Profile Completeness
if "default_profile" in df.columns and "default_profile_image" in df.columns:
    df["profile_complete"] = (
        (df["default_profile"] == False) &
        (df["default_profile_image"] == False)
    ).astype(int)
else:
    df["profile_complete"] = 0


# ==============================
# 4️⃣ Convert Boolean Columns
# ==============================
bool_cols = df.select_dtypes(include="bool").columns
df[bool_cols] = df[bool_cols].astype(int)


# ==============================
# 5️⃣ Select Important Features
# ==============================
features = [
    "followers_count",
    "friends_count",
    "statuses_count",
    "favourites_count",
    "listed_count",
    "followers_friends_ratio",
    "account_age_days",
    "activity_rate",
    "profile_complete",
    
]

# Keep only available columns
features = [col for col in features if col in df.columns]

X = df[features]
y = df["label"]

print("Final Feature Shape:", X.shape)


# ==============================
# 6️⃣ Feature Scaling
# ==============================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ==============================
# 7️⃣ Save Processed Files
# ==============================
pd.DataFrame(X_scaled, columns=features).to_csv("X_processed.csv", index=False)
y.to_csv("y_labels.csv", index=False)

joblib.dump(scaler, "scaler.pkl")

print("Preprocessing Completed Successfully.")
print("Files Saved:")
print(" - X_processed.csv")
print(" - y_labels.csv")
print(" - scaler.pkl")

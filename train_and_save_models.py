# train_and_save_models.py
import pickle
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd

# Load your cleaned/balanced dataset
df = pd.read_csv("balanced_train_dataset.csv")

X = df.drop("label", axis=1)
y = df["label"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train models
rf = RandomForestClassifier()
rf.fit(X_train_scaled, y_train)



# Old way (triggers warning)
# xgb = XGBClassifier(use_label_encoder=False)

# Correct way (no warning)
xgb = XGBClassifier(
    eval_metric='logloss',   # required instead of use_label_encoder
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)


svm = SVC(probability=True)
svm.fit(X_train_scaled, y_train)

# Save scaler and models correctly
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("rf_model.pkl", "wb") as f:
    pickle.dump(rf, f)

with open("xgb_model.pkl", "wb") as f:
    pickle.dump(xgb, f)

with open("svm_model.pkl", "wb") as f:
    pickle.dump(svm, f)

print("Scaler and all models saved successfully!")

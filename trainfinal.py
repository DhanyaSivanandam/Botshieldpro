import pickle
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd

# Load preprocessed dataset
df = pd.read_csv("balanced_train_dataset.csv")

X = df.drop("label", axis=1)
y = df["label"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Train Random Forest
rf = RandomForestClassifier()
rf.fit(X_train_scaled, y_train)
with open("rf_model.pkl", "wb") as f:
    pickle.dump(rf, f)

# Train XGBoost
xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
xgb.fit(X_train_scaled, y_train)
with open("xgb_model.pkl", "wb") as f:
    pickle.dump(xgb, f)

# Train SVM
svm = SVC(probability=True)
svm.fit(X_train_scaled, y_train)
with open("svm_model.pkl", "wb") as f:
    pickle.dump(svm, f)

print("All models and scaler saved successfully!")

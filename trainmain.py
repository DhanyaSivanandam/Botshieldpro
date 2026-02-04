import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# 1️⃣ Load dataset
df = pd.read_csv("balanced_train_dataset.csv")  # update path if needed
print("Dataset Shape:", df.shape)

# 2️⃣ Split features and label
X = df.drop(columns=["label"])
y = df["label"]

# 3️⃣ Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Training size:", X_train.shape)
print("Testing size:", X_test.shape)

# 4️⃣ Scale features (required for SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, "scaler.pkl")  # Save scaler

# 5️⃣ Initialize models
rf = RandomForestClassifier(n_estimators=200, random_state=42)
xgb = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    eval_metric='logloss',  # replace use_label_encoder
    random_state=42
)
svm = SVC(probability=True, random_state=42)  # use scaled features

# 6️⃣ Train models
print("Training Random Forest...")
rf.fit(X_train, y_train)
print("Random Forest trained!")

print("Training XGBoost...")
xgb.fit(X_train, y_train)
print("XGBoost trained!")

print("Training SVM...")
svm.fit(X_train_scaled, y_train)
print("SVM trained!")

# 7️⃣ Save models
joblib.dump(rf, "rf_model.pkl")
joblib.dump(xgb, "xgb_model.pkl")
joblib.dump(svm, "svm_model.pkl")
print("Models saved successfully!")

# 8️⃣ Evaluate models
def evaluate_model(model, X_test_input, y_true, name):
    y_pred = model.predict(X_test_input)
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    cr = classification_report(y_true, y_pred)
    print(f"\n{name} Evaluation:")
    print("Accuracy:", acc)
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", cr)

evaluate_model(rf, X_test, y_test, "Random Forest")
evaluate_model(xgb, X_test, y_test, "XGBoost")
evaluate_model(svm, X_test_scaled, y_test, "SVM")

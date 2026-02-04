import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 1️⃣ Load balanced dataset
df = pd.read_csv("balanced_train_dataset.csv")

print("Dataset Shape:", df.shape)

# 2️⃣ Separate Features and Label
X = df.drop("label", axis=1)
y = df["label"]

# 3️⃣ Train-Test Split (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training size:", X_train.shape)
print("Testing size:", X_test.shape)

# 4️⃣ Train Random Forest
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

# 5️⃣ Predict
y_pred = rf_model.predict(X_test)

# 6️⃣ Evaluation
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

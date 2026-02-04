import pickle
import numpy as np

# Load scaler and models
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("rf_model.pkl", "rb") as f:
    rf = pickle.load(f)

with open("xgb_model.pkl", "rb") as f:
    xgb = pickle.load(f)

with open("svm_model.pkl", "rb") as f:
    svm = pickle.load(f)

# Input feature values (example)
account_features = [
50,20,300,5,0,1,1,0,0,0,0,0,0,0,0,0,0,1,10,0.066,0.1
]


# Convert to numpy array and scale
X_input = np.array(account_features).reshape(1, -1)
X_input_scaled = scaler.transform(X_input)

# Predict probabilities
rf_prob = rf.predict_proba(X_input_scaled)[0][1]
xgb_prob = xgb.predict_proba(X_input_scaled)[0][1]
svm_prob = svm.predict_proba(X_input_scaled)[0][1]

# Compute average risk score
risk_score = (rf_prob + xgb_prob + svm_prob) / 3

# Decide label
label = "Fake" if risk_score < 0.5 else "Genunine"

print(f"Risk Score: {risk_score:.4f}")
print(f"Account Prediction: {label}")

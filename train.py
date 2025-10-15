import os
import pandas as pd
import xgboost as xgb
import joblib

DATA_DIR = 'data'
TRAINED_MODEL_DIR = 'models'
TRAIN_PATH = os.path.join(DATA_DIR, 'train.csv')
XGB_MODEL_PATH = os.path.join(TRAINED_MODEL_DIR, 'model.pkl')
TARGET_COL = 'company_size'

# Load training data
df = pd.read_csv(TRAIN_PATH)

# Separate features and target
X = df.drop(TARGET_COL, axis=1)
y = df[TARGET_COL]



# Train XGBoost Classifier
xgb_clf = xgb.XGBClassifier(n_estimators=100, random_state=42)
xgb_clf.fit(X, y)

if not os.path.exists(TRAINED_MODEL_DIR):
    os.makedirs(TRAINED_MODEL_DIR)

joblib.dump(xgb_clf, XGB_MODEL_PATH)
print(f"XGBoost model trained and saved to {XGB_MODEL_PATH}")

import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import joblib

DATA_DIR = 'data'
TRAINED_MODEL_DIR = 'models'
TRAIN_PATH = os.path.join(DATA_DIR, 'train.csv')
LOG_MODEL_PATH = os.path.join(TRAINED_MODEL_DIR, 'logistic_model.pkl')
TARGET_COL = 'company_size'

# Load training data
df = pd.read_csv(TRAIN_PATH)

# Separate features and target
X = df.drop(TARGET_COL, axis=1)
y = df[TARGET_COL]

# Train Logistic Regression model
log_reg = LogisticRegression(max_iter=200)
log_reg.fit(X, y)

if not os.path.exists(TRAINED_MODEL_DIR):
    os.makedirs(TRAINED_MODEL_DIR)

joblib.dump(log_reg, LOG_MODEL_PATH)
print(f"Logistic Regression model trained and saved to {LOG_MODEL_PATH}")

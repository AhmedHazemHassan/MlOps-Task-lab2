import os
import pandas as pd
import joblib
from sklearn.metrics import  confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json


DATA_DIR = 'data'
TRAINED_MODEL_DIR = 'models'
RESULT_DIR = 'results'
TEST_PATH = os.path.join(DATA_DIR, 'test.csv')
MODEL_PATH = os.path.join(TRAINED_MODEL_DIR, 'model.pkl')
REPORT_PATH = os.path.join(RESULT_DIR, 'accuracy.json')
CONF_MATRIX_PATH = os.path.join(RESULT_DIR, 'confusion_matrix.png')
TARGET_COL = 'company_size'
CONF_SUBSET = 5  # Number of classes to show in confusion matrix plot

# Model accuracy only
# Load test data
df = pd.read_csv(TEST_PATH)
X_test = df.drop(TARGET_COL, axis=1)
y_test = df[TARGET_COL]


# Load trained logistic regression model
log_reg = joblib.load(MODEL_PATH)

# Predict
y_pred = log_reg.predict(X_test)

# Ensure results directory exists
if not os.path.exists(RESULT_DIR):
	os.makedirs(RESULT_DIR)
# Model accuracy only
accuracy = (y_pred == y_test).mean()
with open(REPORT_PATH, 'w') as f:
	json.dump({'accuracy': accuracy}, f, indent=2)
print(f"Model accuracy saved to {REPORT_PATH}")


# Confusion matrix plot (all classes)
import numpy as np
labels = np.unique(y_test)
cm = confusion_matrix(y_test, y_pred, labels=labels)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix (All Classes)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig(CONF_MATRIX_PATH)
plt.close()
print(f"Confusion matrix plot (all classes) saved to {CONF_MATRIX_PATH}")

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DATA_DIR = 'data'
DATASET_FILE = 'salaries.csv'
INPUT_PATH = os.path.join(DATA_DIR, DATASET_FILE)
TRAIN_PATH = os.path.join(DATA_DIR, 'train.csv')
TEST_PATH = os.path.join(DATA_DIR, 'test.csv')

# Step 1: Read salaries CSV file
df = pd.read_csv(INPUT_PATH)

# Step 2: Drop missing values
df = df.dropna()

# Step 3: Encode categorical features
one_hot_cols = ['experience_level', 'employment_type', 'company_size']
label_cols = ['salary_currency', 'employee_residence', 'company_location']
target_col = 'job_title'  # Change if your target column name is different

# One-hot encoding for low-cardinality features
df = pd.get_dummies(df, columns=one_hot_cols)

# Label encoding for high-cardinality features
le_dict = {}
for col in label_cols:
	le = LabelEncoder()
	df[col] = le.fit_transform(df[col])
	le_dict[col] = le


# Label encode the target column
le_target = LabelEncoder()
df[target_col] = le_target.fit_transform(df[target_col])
le_dict[target_col] = le_target

# Remove rare job titles (classes with only one sample)
vc = df[target_col].value_counts()
rare_labels = vc[vc < 2].index
df = df[~df[target_col].isin(rare_labels)]

# Step 4: Train/test split (70/30)
train_df, test_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df[target_col])

# Step 5: Output train and test CSVs
train_df.to_csv(TRAIN_PATH, index=False)
test_df.to_csv(TEST_PATH, index=False)
print(f"Train and test files saved to '{DATA_DIR}' folder.")

import pandas as pd
import os

# URL for the Iris dataset
url = 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv'
df = pd.read_csv(url)

# Ensure the data directory exists
os.makedirs('data', exist_ok=True)

# Save the raw data
df.to_csv('data/data_raw.csv', index=False)
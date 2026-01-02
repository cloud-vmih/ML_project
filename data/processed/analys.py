import pandas as pd
import os

print("Current working directory:", os.getcwd())
data = pd.read_csv('data/processed/clean_movies_data_1975_2025.csv')
data.info()
data.describe()
import pandas as pd

# Load the dataset
df = pd.read_csv("data/raw/Nepali_news_dataset.csv")

# Show the first 5 rows
print(df.head())

# Show dataset information
print("\nDataset Info:")
print(df.info())

# Show column names
print("\nColumns:")
print(df.columns)

# Show dataset size
print("\nShape:")
print(df.shape)
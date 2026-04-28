import pandas as pd

df = pd.read_parquet("data/civil.parquet")

print("Shape:", df.shape)
print("\nColumns:")
print(df.columns)

print("\nFirst rows:")
print(df.head())

print("\nSample row:")
print(df.iloc[0])
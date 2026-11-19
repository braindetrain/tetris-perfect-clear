import pandas as pd

# Load the dataset
df = pd.read_csv("data/tetris_perfect_clear_sequences.csv")


# Display the first five rows
print("Dataset preview:")
print(df.head())

# Display dataset dimensions
print("\nDataset dimensions:")
print(df.shape)

# Display column information
print("\nDataset information:")
print(df.info())

# Count successful and unsuccessful sequences
print("\nSuccess counts:")
print(df["success"].value_counts())

# Calculate success rate
success_rate = df["success"].mean() * 100

print("\nSuccess rate:")
print(f"{success_rate:.2f}%")

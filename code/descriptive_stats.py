import os
import pandas as pd
import matplotlib.pyplot as plt

# Create output folders if they don't exist
os.makedirs("docs/figures", exist_ok=True)
os.makedirs("docs/tables", exist_ok=True)

# Load the dataset
df = pd.read_csv("data/tetris_perfect_clear_sequences.csv")

# Numeric columns, excluding the success outcome
numeric_cols = [c for c in df.select_dtypes(include="number").columns if c != "success"]

# Table 1: summary statistics for numeric columns
if numeric_cols:
    stats = df[numeric_cols].describe().T
    stats["median"] = df[numeric_cols].median()
    stats["iqr"] = df[numeric_cols].quantile(0.75) - df[numeric_cols].quantile(0.25)
    stats.to_csv("docs/tables/numeric_summary.csv")

# Table 2: success counts and percentages
counts = df["success"].value_counts()
percent = (df["success"].value_counts(normalize=True) * 100).round(2)
pd.DataFrame({"count": counts, "percent": percent}).to_csv("docs/tables/success_counts.csv")

# Graph 1: success vs. unsuccessful bar chart
counts.plot(kind="bar")
plt.title("Successful vs. Unsuccessful Sequences")
plt.xlabel("Success")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("docs/figures/success_counts.png")
plt.close()

# Graph 2: histogram of every numeric column
for col in numeric_cols:
    df[col].hist()
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"docs/figures/{col}_hist.png")
    plt.close()

# Graph 3: boxplot of each numeric column split by success
for col in numeric_cols:
    df.boxplot(column=col, by="success")
    plt.title(f"{col} by Success")
    plt.suptitle("")
    plt.tight_layout()
    plt.savefig(f"docs/figures/{col}_by_success.png")
    plt.close()

print("Saved tables to docs/tables/ and figures to docs/figures/")
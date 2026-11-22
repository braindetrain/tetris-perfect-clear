import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs("docs/figures", exist_ok=True)
os.makedirs("docs/tables", exist_ok=True)

df = pd.read_csv("data/tetris_perfect_clear_sequences.csv")

# Table 1: stats for numeric columns
numeric_stats = df.describe().T
numeric_stats["median"] = df.median(numeric_only=True)
numeric_stats.to_csv("docs/tables/numeric_summary.csv")

# Table 2: success counts and percentages
success_counts = df["success"].value_counts()
success_pct = df["success"].value_counts(normalize=True) * 100
pd.DataFrame({"count": success_counts, "percent": success_pct.round(2)}) \
  .to_csv("docs/tables/success_counts.csv")

# Graph 1: success vs. failure bar chart
success_counts.plot(kind="bar")
plt.title("Successful vs. Unsuccessful Sequences")
plt.xlabel("Success")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("docs/figures/success_counts.png")
plt.close()

# Graph 2: histogram of a numeric column
df["[some_numeric_column]"].hist()
plt.title("Distribution of [some_numeric_column]")
plt.savefig("docs/figures/[column]_hist.png")
plt.close()
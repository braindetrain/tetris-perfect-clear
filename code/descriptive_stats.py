import os
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("docs/figures", exist_ok=True)
os.makedirs("docs/tables", exist_ok=True)

df = pd.read_csv("data/tetris_perfect_clear_sequences.csv")
position_cols = ["piece_1", "piece_2", "piece_3", "piece_4"]

# Table 1 + Graph 1: overall success counts and percentages (Question 1)
counts = df["success"].value_counts().sort_index()
percent = (df["success"].value_counts(normalize=True).sort_index() * 100).round(2)
pd.DataFrame({"count": counts, "percent": percent}).to_csv("docs/tables/success_counts.csv")

counts.plot(kind="bar")
plt.title("Sequences That Allow a Perfect Clear (1) vs. Not (0)")
plt.xlabel("Success")
plt.ylabel("Number of sequences")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("docs/figures/success_counts.png")
plt.close()

# Table 2: success rate for each piece at each position (Question 2)
rows = []
for col in position_cols:
    grouped = df.groupby(col)["success"].agg(["count", "sum", "mean"])
    for piece, r in grouped.iterrows():
        rows.append({"position": col, "piece": piece, "sequences": int(r["count"]),
                     "successes": int(r["sum"]), "success_rate_pct": round(r["mean"] * 100, 2)})
by_position = pd.DataFrame(rows)
by_position.to_csv("docs/tables/success_by_piece_and_position.csv", index=False)

# Graph 2: grouped bar chart of those success rates
chart = by_position.pivot(index="piece", columns="position", values="success_rate_pct")
chart.plot(kind="bar", figsize=(9, 5))
plt.title("Success Rate by Piece and Position")
plt.xlabel("Piece")
plt.ylabel("Success rate (%)")
plt.ylim(0, 100)
plt.xticks(rotation=0)
plt.legend(title="Position")
plt.tight_layout()
plt.savefig("docs/figures/success_rate_by_piece_position.png")
plt.close()

# Table 3 + Graph 3: success rate for each piece_1 / piece_2 pair (Question 3)
pair = df.pivot_table(index="piece_1", columns="piece_2", values="success", aggfunc="mean") * 100
pair.round(1).to_csv("docs/tables/success_rate_piece1_piece2.csv")

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(pair.values, cmap="viridis", vmin=0, vmax=100)
ax.set_xticks(range(len(pair.columns)))
ax.set_xticklabels(pair.columns)
ax.set_yticks(range(len(pair.index)))
ax.set_yticklabels(pair.index)
ax.set_xlabel("Piece 2")
ax.set_ylabel("Piece 1")
ax.set_title("Success Rate (%) by First Two Pieces")
fig.colorbar(im, label="Success rate (%)")
plt.tight_layout()
plt.savefig("docs/figures/success_rate_heatmap.png")
plt.close()

print("Saved tables to docs/tables/ and figures to docs/figures/")
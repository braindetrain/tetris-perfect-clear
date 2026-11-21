import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

df = pd.read_csv("data/tetris_perfect_clear_sequences.csv")
y = df["success"]

# Shuffle rows before splitting so each fold is a mix of sequences
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Baseline features: one-hot piece at each position
X1 = pd.get_dummies(df[["piece_1", "piece_2", "piece_3", "piece_4"]])

# Added feature: the unordered pair of the first two pieces (e.g. "OZ")
df["first_two"] = df[["piece_1", "piece_2"]].apply(lambda r: "".join(sorted(r)), axis=1)
X2 = pd.get_dummies(df[["piece_1", "piece_2", "piece_3", "piece_4", "first_two"]])

print(f"Majority-class baseline: {y.mean():.4f}")
for depth in [3, 5, 8, 12, None]:
    tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
    s1 = cross_val_score(tree, X1, y, cv=cv).mean()
    s2 = cross_val_score(tree, X2, y, cv=cv).mean()
    print(f"depth {depth}: one-hot only {s1:.4f} | with pair feature {s2:.4f}")
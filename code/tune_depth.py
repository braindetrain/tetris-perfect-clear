import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

df = pd.read_csv("data/tetris_perfect_clear_sequences.csv")
X = pd.get_dummies(df[["piece_1", "piece_2", "piece_3", "piece_4"]])
y = df["success"]

# Compare tree depths using 5-fold cross-validation (no test-set peeking)
for depth in [3, 5, 8, 12, None]:
    scores = cross_val_score(DecisionTreeClassifier(max_depth=depth, random_state=42), X, y, cv=5)
    print(depth, round(scores.mean(), 4))
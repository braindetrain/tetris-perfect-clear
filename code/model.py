import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay

os.makedirs("docs/figures", exist_ok=True)
os.makedirs("docs/tables", exist_ok=True)

df = pd.read_csv("data/tetris_perfect_clear_sequences.csv")

# Feature engineering: unordered pair of the first two pieces (e.g. "OZ").
# Motivated by the descriptive results: success depends on specific piece pairs.
df["first_two"] = df[["piece_1", "piece_2"]].apply(lambda r: "".join(sorted(r)), axis=1)

# One-hot encode piece positions plus the pair feature
X = pd.get_dummies(df[["piece_1", "piece_2", "piece_3", "piece_4", "first_two"]])
y = df["success"]

# 80/20 stratified split keeps the success ratio the same in train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# max_depth=5 chosen from cross-validation (code/tune_depth.py)
model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
baseline = y_test.value_counts(normalize=True).max()
print(f"Train accuracy: {model.score(X_train, y_train):.4f}")
print(f"Decision tree accuracy: {accuracy:.4f}")
print(f"Majority-class baseline: {baseline:.4f}")

# Cross-validated accuracy is a steadier estimate than one small test split
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_acc = cross_val_score(DecisionTreeClassifier(max_depth=5, random_state=42), X, y, cv=cv).mean()
print(f"5-fold cross-validated accuracy: {cv_acc:.4f}")

pd.DataFrame({
    "metric": ["tree_accuracy", "majority_baseline", "cv_accuracy"],
    "value": [round(accuracy, 4), round(baseline, 4), round(cv_acc, 4)],
}).to_csv("docs/tables/model_metrics.csv", index=False)

ConfusionMatrixDisplay.from_predictions(y_test, predictions)
plt.title("Decision Tree Confusion Matrix")
plt.tight_layout()
plt.savefig("docs/figures/confusion_matrix.png")
plt.close()

importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
importance.to_csv("docs/tables/feature_importance.csv", header=["importance"])

importance.head(10).sort_values().plot(kind="barh")
plt.title("Top 10 Feature Importances")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("docs/figures/feature_importance.png")
plt.close()

print("Saved model outputs to docs/")
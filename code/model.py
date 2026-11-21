import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay

os.makedirs("docs/figures", exist_ok=True)
os.makedirs("docs/tables", exist_ok=True)

df = pd.read_csv("data/tetris_perfect_clear_sequences.csv")

# Preparation: one-hot encode the piece columns so the tree can use them
X = pd.get_dummies(df[["piece_1", "piece_2", "piece_3", "piece_4"]])
y = df["success"]

# 80/20 split; stratify keeps the success/failure ratio the same in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train the decision tree (max_depth limits overfitting)
model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print(f"Train accuracy: {model.score(X_train, y_train):.4f}")

# Performance metric: accuracy, compared with a majority-class baseline
accuracy = accuracy_score(y_test, predictions)
baseline = y_test.value_counts(normalize=True).max()
print(f"Decision tree accuracy: {accuracy:.4f}")
print(f"Majority-class baseline: {baseline:.4f}")

pd.DataFrame({"metric": ["tree_accuracy", "majority_baseline"],
              "value": [round(accuracy, 4), round(baseline, 4)]}) \
  .to_csv("docs/tables/model_metrics.csv", index=False)

# Confusion matrix figure
ConfusionMatrixDisplay.from_predictions(y_test, predictions)
plt.title("Decision Tree Confusion Matrix")
plt.tight_layout()
plt.savefig("docs/figures/confusion_matrix.png")
plt.close()

# Feature importance table and figure
importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
importance.to_csv("docs/tables/feature_importance.csv", header=["importance"])

importance.head(10).sort_values().plot(kind="barh")
plt.title("Top 10 Feature Importances")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("docs/figures/feature_importance.png")
plt.close()

print("Saved model outputs to docs/")
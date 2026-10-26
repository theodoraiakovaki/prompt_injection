import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline


# 1. Load the existing training and validation splits.
def load_split(split_name):
    path = Path("data") / f"{split_name}.json"

    with path.open("r", encoding="utf-8") as file:
        rows = json.load(file)

    texts = [row["prompt"] for row in rows]
    labels = [row["label"] for row in rows]

    return texts, labels


X_train, y_train = load_split("train")
X_val, y_val = load_split("validation")

# 2. Connect text representation and classification.
model = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=20_000)),
    ("classifier", LogisticRegression(max_iter=1000)),
])

# 3. Learn from the training examples.
print("Training...")
model.fit(X_train, y_train)

# 4. Predict labels for the validation examples.
predictions = model.predict(X_val)

# 5. Compare predictions with the known validation labels.
print("\nValidation results:")
print(classification_report(
    y_val,
    predictions,
    labels=[0, 1],
    target_names=["benign", "injection"],
    digits=3,
    zero_division=0,
))

matrix = confusion_matrix(y_val, predictions, labels=[0, 1])

print("Confusion matrix:")
print("Order: benign, injection")
print(matrix)

tn, fp, fn, tp = matrix.ravel()
false_positive_rate = fp / (fp + tn)

print(f"\nFalse-positive rate: {false_positive_rate:.2%}")
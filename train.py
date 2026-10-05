"""
Train a sentiment model on the Amazon reviews dataset and save it as model.pkl.

Run:
    python train.py
"""
import csv
import os

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

DATA_PATH = os.path.join(os.path.dirname(__file__), "datasets", "amazon_dataset.csv")

texts, labels = [], []
with open(DATA_PATH, encoding="utf-8") as f:
    for row in csv.DictReader(f):
        texts.append(row["reviewText"])
        labels.append(int(row["Positive"]))  # 1 = positive, 0 = negative

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

# TF-IDF turns text into numbers; logistic regression classifies them.
# ngram_range=(1, 2) lets the model see phrases like "not good".
model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
])
model.fit(X_train, y_train)

acc = accuracy_score(y_test, model.predict(X_test))
print(f"Test accuracy: {acc:.3f}")

joblib.dump(model, os.path.join(os.path.dirname(__file__), "model.pkl"))
print("Saved model to model.pkl")

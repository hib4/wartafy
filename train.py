import os
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from data_loader import load_ag_news

MODEL_DIR = "models"


def build_pipelines() -> dict:
    return {
        "Logistic Regression": Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=50_000,
                ngram_range=(1, 2),
                sublinear_tf=True,
                min_df=2,
            )),
            ("clf", LogisticRegression(
                max_iter=1000,
                C=5.0,
                solver="lbfgs",
                multi_class="auto",
            )),
        ]),
        "Naive Bayes": Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=50_000,
                ngram_range=(1, 2),
                sublinear_tf=True,
                min_df=2,
            )),
            ("clf", MultinomialNB(alpha=0.1)),
        ]),
    }


def train_and_save(train_df: pd.DataFrame) -> dict:
    os.makedirs(MODEL_DIR, exist_ok=True)

    X_train = train_df["clean_text"]
    y_train = train_df["category"]

    pipelines = build_pipelines()
    trained = {}

    for name, pipeline in pipelines.items():
        print(f"Training {name}...")
        pipeline.fit(X_train, y_train)
        safe_name = name.lower().replace(" ", "_")
        path = os.path.join(MODEL_DIR, f"{safe_name}.pkl")
        with open(path, "wb") as f:
            pickle.dump(pipeline, f)
        print(f"  Saved -> {path}")

        trained[name] = pipeline

    return trained


def load_models() -> dict:
    models = {}
    for name, safe in [("Logistic Regression", "logistic_regression"),
                        ("Naive Bayes",         "naive_bayes")]:
        path = os.path.join(MODEL_DIR, f"{safe}.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                models[name] = pickle.load(f)
    return models


if __name__ == "__main__":
    train_df, _ = load_ag_news(max_train=20_000, max_test=2_000)
    trained = train_and_save(train_df)
    print("\nAll models trained and saved successfully.")
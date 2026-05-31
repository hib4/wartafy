import os
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from data_loader import load_ag_news

MODEL_DIR = "models"


class LabelDecodingPipeline:
    """Wraps a sklearn Pipeline trained on integer labels to return string predictions."""

    def __init__(self, pipeline, int_to_label: dict, classes: list):
        self.pipeline = pipeline
        self.int_to_label = int_to_label
        self.classes_ = classes  # string class names for predict_proba alignment

    def predict(self, X):
        preds = self.pipeline.predict(X)
        return [self.int_to_label[p] for p in preds]

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)


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
        "Neural Network": Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=50_000,
                ngram_range=(1, 2),
                sublinear_tf=True,
                min_df=2,
            )),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                solver="adam",
                max_iter=300,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=42,
                verbose=False,
            )),
        ]),
        "XGBoost": Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=50_000,
                ngram_range=(1, 2),
                sublinear_tf=True,
                min_df=2,
            )),
            ("clf", XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                objective="multi:softprob",
                num_class=4,
                n_jobs=-1,
                random_state=42,
                use_label_encoder=False,
                verbosity=0,
            )),
        ]),
    }


def train_and_save(train_df: pd.DataFrame) -> dict:
    os.makedirs(MODEL_DIR, exist_ok=True)

    X_train = train_df["clean_text"]
    y_train_str = train_df["category"]
    unique_cats = list(y_train_str.unique())
    label_to_int = {cat: i for i, cat in enumerate(unique_cats)}
    int_to_label = {i: cat for cat, i in label_to_int.items()}
    y_train = y_train_str.map(label_to_int)
    classes = sorted(int_to_label.values())

    pipelines = build_pipelines()
    trained = {}

    for name, pipeline in pipelines.items():
        print(f"Training {name}...")
        pipeline.fit(X_train, y_train)

        if pipeline.classes_[0] in int_to_label:
            wrapper = LabelDecodingPipeline(pipeline, int_to_label, classes)
        else:
            wrapper = pipeline

        safe_name = name.lower().replace(" ", "_")
        path = os.path.join(MODEL_DIR, f"{safe_name}.pkl")
        with open(path, "wb") as f:
            pickle.dump(wrapper, f)
        print(f"  Saved -> {path}")

        trained[name] = wrapper

    return trained


def load_models() -> dict:
    models = {}
    for name, safe in [("Logistic Regression", "logistic_regression"),
                        ("Naive Bayes",         "naive_bayes"),
                        ("Neural Network",      "neural_network"),
                        ("XGBoost",             "xgboost")]:
        path = os.path.join(MODEL_DIR, f"{safe}.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                models[name] = pickle.load(f)
    return models


if __name__ == "__main__":
    train_df, _ = load_ag_news(max_train=20_000, max_test=2_000)
    trained = train_and_save(train_df)
    print("\nAll models trained and saved successfully.")
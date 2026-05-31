import pandas as pd
from preprocess import clean_text
import os

LABEL_MAP = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Science/Technology",
}

_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def load_ag_news(max_train: int = 20000, max_test: int = 2000) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("Loading AG News dataset from local files...")

    train_df = _load_csv(os.path.join(_DATA_DIR, "train.csv"), max_train)
    test_df  = _load_csv(os.path.join(_DATA_DIR, "test.csv"),  max_test)

    print(f"  Train samples : {len(train_df):,}")
    print(f"  Test samples  : {len(test_df):,}")
    print(f"  Categories    : {list(LABEL_MAP.values())}\n")

    return train_df, test_df


def _load_csv(path: str, max_samples: int) -> pd.DataFrame:
    df = pd.read_csv(path).head(max_samples).copy()
    df["category"] = df["label"].map(LABEL_MAP)
    print(f"  Preprocessing text ({os.path.basename(path)})...")
    df["clean_text"] = df["text"].apply(clean_text)

    return df[["text", "clean_text", "label", "category"]]
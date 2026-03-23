import pandas as pd
from datasets import load_dataset
from preprocess import clean_text

LABEL_MAP = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Science/Technology",
}


def load_ag_news(max_train: int = 20000, max_test: int = 2000) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("Downloading AG News dataset from HuggingFace...")
    dataset = load_dataset("fancyzhx/ag_news")

    train_df = _to_dataframe(dataset["train"], max_train)
    test_df  = _to_dataframe(dataset["test"],  max_test)

    print(f"  Train samples : {len(train_df):,}")
    print(f"  Test samples  : {len(test_df):,}")
    print(f"  Categories    : {list(LABEL_MAP.values())}\n")

    return train_df, test_df


def _to_dataframe(split, max_samples: int) -> pd.DataFrame:
    df = split.to_pandas().head(max_samples).copy()
    df["text"] = df["text"]
    df["category"] = df["label"].map(LABEL_MAP)
    print("  Preprocessing text...")
    df["clean_text"] = df["text"].apply(clean_text)

    return df[["text", "clean_text", "label", "category"]]
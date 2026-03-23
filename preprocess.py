import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def download_nltk_resources():
    resources = ["stopwords", "punkt", "punkt_tab"]
    for resource in resources:
        try:
            nltk.data.find(f"tokenizers/{resource}" if "punkt" in resource else f"corpora/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)

download_nltk_resources()

STOP_WORDS = set(stopwords.words("english"))


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]

    return " ".join(tokens)
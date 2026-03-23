"""
preprocess.py
=============
Handles all text cleaning and preprocessing for the Fake News Detection System.
Steps: lowercase → remove noise → tokenize → remove stopwords → lemmatize
"""

import re
import string
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# ── Download required NLTK assets (runs once) ────────────────────────────────
def download_nltk_resources():
    resources = ['punkt', 'stopwords', 'wordnet', 'omw-1.4', 'punkt_tab']
    for r in resources:
        nltk.download(r, quiet=True)

download_nltk_resources()

# ── Core cleaning function ────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """
    Full pipeline: lowercase → strip URLs/HTML/punctuation →
    tokenize → remove stopwords → lemmatize → rejoin.
    Returns a clean string ready for TF-IDF.
    """
    if not isinstance(text, str):
        return ""

    lemmatizer = WordNetLemmatizer()
    stop_words  = set(stopwords.words('english'))

    # 1. Lowercase
    text = text.lower()

    # 2. Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # 3. Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # 4. Remove punctuation and digits
    text = text.translate(str.maketrans('', '', string.punctuation + string.digits))

    # 5. Tokenize
    tokens = word_tokenize(text)

    # 6. Remove stopwords and short tokens
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]

    # 7. Lemmatize
    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return " ".join(tokens)


# ── Dataset loader ────────────────────────────────────────────────────────────
def load_and_prepare_data(fake_path: str, true_path: str) -> pd.DataFrame:
    """
    Loads the Kaggle Fake/True CSV files, adds labels, merges,
    cleans text, and returns a ready-to-train DataFrame.

    Kaggle dataset: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
      Fake.csv  → label 0
      True.csv  → label 1
    """
    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)

    fake_df['label'] = 0   # 0 = Fake
    true_df['label'] = 1   # 1 = Real

    df = pd.concat([fake_df, true_df], ignore_index=True)

    # Combine title + text for richer features
    df['content'] = (
        df.get('title', pd.Series([''] * len(df))).fillna('') +
        ' ' +
        df.get('text',  pd.Series([''] * len(df))).fillna('')
    )

    print(f"[preprocess] Loaded {len(df)} articles  "
          f"(Fake: {(df.label==0).sum()} | Real: {(df.label==1).sum()})")

    print("[preprocess] Cleaning text… (this may take a minute)")
    df['cleaned'] = df['content'].apply(clean_text)

    # Drop rows where cleaning produced an empty string
    df = df[df['cleaned'].str.strip() != ''].reset_index(drop=True)

    print(f"[preprocess] Done. {len(df)} usable articles after cleaning.")
    return df[['cleaned', 'label']]


# ── Quick smoke-test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = "BREAKING NEWS: Scientists Discover That Water Is Wet!! Visit https://fake.com for more."
    print("Original :", sample)
    print("Cleaned  :", clean_text(sample))
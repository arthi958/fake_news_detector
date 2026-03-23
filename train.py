"""
train.py
========
Trains two classifiers (Logistic Regression + Naive Bayes) on TF-IDF features,
evaluates both, saves the best model + vectorizer, and prints a comparison table.

Run:
    python train.py
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')          # headless – safe for server environments
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)

from preprocess import load_and_prepare_data, clean_text

warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR   = "data"
MODEL_DIR  = "models"
FAKE_CSV   = os.path.join(DATA_DIR, "Fake.csv")
TRUE_CSV   = os.path.join(DATA_DIR, "True.csv")
os.makedirs(MODEL_DIR, exist_ok=True)

# ── 1. Load & preprocess ──────────────────────────────────────────────────────
def get_data():
    """Return a preprocessed DataFrame, using a small synthetic set if CSVs are absent."""
    if os.path.exists(FAKE_CSV) and os.path.exists(TRUE_CSV):
        return load_and_prepare_data(FAKE_CSV, TRUE_CSV)

    # ── Synthetic fallback (demo / CI) ────────────────────────────────────────
    print("[train] Kaggle CSVs not found → generating synthetic demo dataset.")
    fake_samples = [
        "government secretly controls weather machines hidden underground bases",
        "celebrity clone conspiracy revealed by insider whistleblower sources",
        "miracle cure cancer discovered suppressed big pharma profits hidden truth",
        "aliens landed white house politicians hiding extraterrestrial contact proof",
        "moon landing hoax nasa studio kubrick filmed fake footage admitted",
        "vaccines cause autism doctors refuse admit evidence overwhelming proof",
        "chemtrails mind control government spraying population chemicals planes",
        "deep state elite bankers controlling world economy shadow government exposed",
        "reptilian shapeshifters running world governments royal family lizard people",
        "election rigged voting machines hacked millions fake ballots stuffed boxes",
        "5g towers spreading covid virus activated microchips population control",
        "hollywood satanic rituals exposed celebrities secret society illuminati members",
        "fluoride water supply causing brain damage population dumbing down intentional",
        "soros funding migrant crisis open borders globalist agenda exposed truth",
        "flat earth nasa hiding truth antarctica ice wall edge planet proof",
    ] * 30

    real_samples = [
        "federal reserve raises interest rates combat inflation economic growth",
        "scientists publish peer reviewed study climate change effects agriculture",
        "congress passes bipartisan infrastructure bill roads bridges funding approved",
        "world health organization releases annual global health statistics report",
        "stock markets close higher technology sector leads gains investors optimistic",
        "university researchers discover new treatment resistant bacterial infections",
        "international climate summit produces agreement carbon emission reductions",
        "supreme court hears arguments landmark civil rights employment discrimination",
        "treasury department releases quarterly economic outlook forecast stable growth",
        "nasa successfully launches new satellite earth observation weather forecasting",
        "pharmaceutical company completes phase three trial new diabetes medication",
        "united nations security council votes resolution peacekeeping operations",
        "federal election commission reports campaign finance contributions candidates",
        "census bureau releases annual poverty income health insurance statistics data",
        "department energy announces renewable energy investment solar wind projects",
    ] * 30

    texts  = fake_samples + real_samples
    labels = [0] * len(fake_samples) + [1] * len(real_samples)
    df = pd.DataFrame({'content': texts, 'label': labels})
    df['cleaned'] = df['content'].apply(clean_text)
    df = df[df['cleaned'].str.strip() != ''].reset_index(drop=True)
    print(f"[train] Synthetic dataset: {len(df)} rows "
          f"(Fake: {(df.label==0).sum()} | Real: {(df.label==1).sum()})")
    return df[['cleaned', 'label']]


# ── 2. Feature extraction ─────────────────────────────────────────────────────
def build_tfidf(X_train, X_test):
    """Fit TF-IDF on train, transform both splits. Returns (vectorizer, Xtr, Xte)."""
    vec = TfidfVectorizer(
        max_features=50_000,   # top 50k tokens
        ngram_range=(1, 2),    # unigrams + bigrams
        sublinear_tf=True,     # apply log(tf) scaling
        min_df=2               # ignore very rare tokens
    )
    Xtr = vec.fit_transform(X_train)
    Xte = vec.transform(X_test)
    print(f"[train] TF-IDF vocabulary size: {len(vec.vocabulary_):,}")
    return vec, Xtr, Xte


# ── 3. Train & evaluate helpers ───────────────────────────────────────────────
def evaluate(name, model, X_test, y_test):
    """Prints metrics, returns dict of results."""
    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Fake", "Real"])
    cm     = confusion_matrix(y_test, y_pred)

    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    print(f"  Accuracy : {acc*100:.2f}%")
    print(f"\n{report}")

    return {"name": name, "model": model, "accuracy": acc,
            "report": report, "cm": cm, "y_pred": y_pred}


def plot_confusion_matrix(results, y_test):
    """Save side-by-side confusion matrices to models/confusion_matrices.png."""
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5))
    if len(results) == 1:
        axes = [axes]

    for ax, res in zip(axes, results):
        disp = ConfusionMatrixDisplay(
            confusion_matrix=res['cm'],
            display_labels=["Fake", "Real"]
        )
        disp.plot(ax=ax, colorbar=False, cmap='Blues')
        ax.set_title(f"{res['name']}\nAccuracy: {res['accuracy']*100:.2f}%",
                     fontsize=13, fontweight='bold')

    plt.tight_layout()
    path = os.path.join(MODEL_DIR, "confusion_matrices.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[train] Confusion matrix saved → {path}")


# ── 4. Main pipeline ──────────────────────────────────────────────────────────
def main():
    # Load data
    df = get_data()

    X = df['cleaned']
    y = df['label']

    # Train / test split  (80 / 20, stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\n[train] Train: {len(X_train)} | Test: {len(X_test)}")

    # TF-IDF
    vectorizer, X_train_tfidf, X_test_tfidf = build_tfidf(X_train, X_test)

    # ── Logistic Regression ────────────────────────────────────────────────────
    print("\n[train] Training Logistic Regression…")
    lr = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs', n_jobs=-1)
    lr.fit(X_train_tfidf, y_train)
    lr_res = evaluate("Logistic Regression", lr, X_test_tfidf, y_test)

    # ── Multinomial Naive Bayes ────────────────────────────────────────────────
    print("\n[train] Training Multinomial Naive Bayes…")
    nb = MultinomialNB(alpha=0.1)
    nb.fit(X_train_tfidf, y_train)
    nb_res = evaluate("Naive Bayes", nb, X_test_tfidf, y_test)

    # ── Comparison table ───────────────────────────────────────────────────────
    results = [lr_res, nb_res]
    print("\n" + "─"*40)
    print("  MODEL COMPARISON")
    print("─"*40)
    for r in results:
        bar = "█" * int(r['accuracy'] * 30)
        print(f"  {r['name']:<25} {r['accuracy']*100:6.2f}%  {bar}")
    print("─"*40)

    best = max(results, key=lambda r: r['accuracy'])
    print(f"\n  ✓ Best model: {best['name']} ({best['accuracy']*100:.2f}%)\n")

    # ── Save best model + vectorizer ───────────────────────────────────────────
    vec_path   = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
    model_path = os.path.join(MODEL_DIR, "best_model.pkl")
    lr_path    = os.path.join(MODEL_DIR, "logistic_regression.pkl")
    nb_path    = os.path.join(MODEL_DIR, "naive_bayes.pkl")

    with open(vec_path,   'wb') as f: pickle.dump(vectorizer, f)
    with open(model_path, 'wb') as f: pickle.dump(best['model'], f)
    with open(lr_path,    'wb') as f: pickle.dump(lr, f)
    with open(nb_path,    'wb') as f: pickle.dump(nb, f)

    print(f"[train] Saved vectorizer  → {vec_path}")
    print(f"[train] Saved best model  → {model_path}")
    print(f"[train] Saved LR model    → {lr_path}")
    print(f"[train] Saved NB model    → {nb_path}")

    # Save metadata (accuracy scores for the UI)
    meta = {
        "lr_accuracy": lr_res['accuracy'],
        "nb_accuracy": nb_res['accuracy'],
        "best_model_name": best['name']
    }
    meta_path = os.path.join(MODEL_DIR, "metadata.pkl")
    with open(meta_path, 'wb') as f: pickle.dump(meta, f)

    # Confusion matrices plot
    plot_confusion_matrix(results, y_test)
    print("\n[train] All artefacts saved. Ready to launch the app!")


if __name__ == "__main__":
    main()
# Fake News Detection System

A complete ML pipeline: NLP preprocessing → TF-IDF → two classifiers → Streamlit UI.

## Project Structure

```
fake_news_detector/
├── preprocess.py        ← Text cleaning & dataset loading
├── train.py             ← Model training, evaluation & saving
├── app.py               ← Streamlit web app
├── requirements.txt     ← Python dependencies
├── README.md
├── data/                ← Place Kaggle CSVs here (optional)
│   ├── Fake.csv
│   └── True.csv
└── models/              ← Auto-created after training
```

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. (Optional) Download Kaggle dataset
https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
Place Fake.csv and True.csv in the data/ folder.
Without it, a synthetic demo dataset is used automatically.

### 3. Train the models
```bash
python train.py
```

### 4. Run the app
```bash
streamlit run app.py
```
Open: http://localhost:8501

## How It Works

1. preprocess.py  — lowercase, remove URLs/HTML/punct, tokenize, remove stopwords, lemmatize
2. train.py       — TF-IDF vectorizer (50k features, bigrams) + train LR and Naive Bayes
3. app.py         — load saved models, run live inference on user input

## Expected Accuracy (Kaggle dataset)
- Logistic Regression: ~98-99%
- Naive Bayes:         ~94-95%

## Troubleshooting
- ModuleNotFoundError: run pip install -r requirements.txt
- Models not found: run python train.py first
- NLTK errors: python -c "import nltk; nltk.download('all')"

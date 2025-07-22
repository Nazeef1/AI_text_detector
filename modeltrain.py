# # AI Text Detector using Hello-SimpleAI/HC3 dataset

# import pandas as pd
# import numpy as np
# import spacy
# import joblib
# import random
# from sklearn.ensemble import RandomForestClassifier, VotingClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# from sklearn.feature_extraction.text import TfidfVectorizer
# from scipy.sparse import hstack, csr_matrix
# from textstat import flesch_reading_ease
# from datasets import load_dataset

# # Load spaCy
# nlp = spacy.load("en_core_web_sm")

# # Load HC3 dataset
# print("Loading HC3 dataset...")
# ds = load_dataset("Hello-SimpleAI/HC3", "all")
# data = ds["train"]

# # Collect all human and AI responses
# human_texts = []
# ai_texts = []

# for row in data:
#     human_texts.extend(row["human_answers"])
#     ai_texts.extend(row["chatgpt_answers"])

# # Match counts
# min_len = min(len(human_texts), len(ai_texts))
# human_texts = human_texts[:min_len]
# ai_texts = ai_texts[:min_len]
# texts = human_texts + ai_texts
# labels = [0] * min_len + [1] * min_len

# # Shuffle the data
# combined = list(zip(texts, labels))
# random.shuffle(combined)
# texts, labels = zip(*combined)

# # POS Feature Extraction
# def extract_pos_stats(doc):
#     pos_counts = doc.count_by(spacy.attrs.POS)
#     total = sum(pos_counts.values())
#     return {
#         "noun_ratio": pos_counts.get(92, 0) / total if total > 0 else 0,
#         "verb_ratio": pos_counts.get(100, 0) / total if total > 0 else 0,
#         "adj_ratio": pos_counts.get(84, 0) / total if total > 0 else 0,
#         "adv_ratio": pos_counts.get(85, 0) / total if total > 0 else 0,
#         "pron_ratio": pos_counts.get(95, 0) / total if total > 0 else 0,
#         "num_ratio": pos_counts.get(93, 0) / total if total > 0 else 0
#     }

# # Extract features
# print("Extracting linguistic features...")
# text_stats = []
# for doc in nlp.pipe(texts, batch_size=64):
#     pos = extract_pos_stats(doc)
#     words = [token.text for token in doc if token.is_alpha]
#     word_count = len(words)
#     avg_word_length = np.mean([len(w) for w in words]) if words else 0
#     char_count = len(doc.text)
#     readability = flesch_reading_ease(doc.text)
#     feats = [
#         pos['noun_ratio'], pos['verb_ratio'], pos['adj_ratio'],
#         pos['adv_ratio'], pos['pron_ratio'], pos['num_ratio'],
#         char_count, word_count, avg_word_length, readability
#     ]
#     text_stats.append(feats)
# text_stats = np.array(text_stats)

# # TF-IDF
# print("Fitting TF-IDF vectorizer...")
# tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
# tfidf_matrix = tfidf_vectorizer.fit_transform(texts)

# # Combine features
# X = hstack([tfidf_matrix, csr_matrix(text_stats)])
# y = np.array(labels)

# # Train-test split
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train model
# print("Training ensemble model...")
# clf = VotingClassifier(estimators=[
#     ('lr', LogisticRegression(max_iter=3000)),
#     ('rf', RandomForestClassifier(n_estimators=100)),
#     ('dt', DecisionTreeClassifier())
# ], voting='soft')
# clf.fit(X_train, y_train)

# # Evaluate
# y_pred = clf.predict(X_val)
# acc = accuracy_score(y_val, y_pred)
# print(f"\nAccuracy: {acc:.4f}")
# print("\nConfusion Matrix:")
# print(confusion_matrix(y_val, y_pred))

# # Save artifacts
# print("Saving model and data...")
# joblib.dump(tfidf_vectorizer, "hc3_tfidf_vectorizer.pkl")
# joblib.dump(clf, "hc3_model.pkl")
# joblib.dump(X_val, "hc3_X_val.pkl")
# joblib.dump(y_val, "hc3_y_val.pkl")
# print("Done.")

# AI Text Detector using Hello-SimpleAI/HC3 dataset

import pandas as pd
import numpy as np
import spacy
import joblib
import random
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
from textstat import flesch_reading_ease
from datasets import load_dataset
from sklearn.pipeline import make_pipeline

# Load spaCy
nlp = spacy.load("en_core_web_sm")

# Load HC3 dataset
print("Loading HC3 dataset...")
ds = load_dataset("Hello-SimpleAI/HC3", "all")
data = ds["train"]

# Collect all human and AI responses
human_texts = []
ai_texts = []

for row in data:
    human_texts.extend(row["human_answers"])
    ai_texts.extend(row["chatgpt_answers"])

# Match counts
min_len = min(len(human_texts), len(ai_texts))
human_texts = human_texts[:min_len]
ai_texts = ai_texts[:min_len]
texts = human_texts + ai_texts
labels = [0] * min_len + [1] * min_len

# Shuffle the data
combined = list(zip(texts, labels))
random.shuffle(combined)
texts, labels = zip(*combined)

# POS Feature Extraction
def extract_pos_stats(doc):
    pos_counts = doc.count_by(spacy.attrs.POS)
    total = sum(pos_counts.values())
    return {
        "noun_ratio": pos_counts.get(92, 0) / total if total > 0 else 0,
        "verb_ratio": pos_counts.get(100, 0) / total if total > 0 else 0,
        "adj_ratio": pos_counts.get(84, 0) / total if total > 0 else 0,
        "adv_ratio": pos_counts.get(85, 0) / total if total > 0 else 0,
        "pron_ratio": pos_counts.get(95, 0) / total if total > 0 else 0,
        "num_ratio": pos_counts.get(93, 0) / total if total > 0 else 0
    }

# Extract features
print("Extracting linguistic features...")
text_stats = []
for doc in nlp.pipe(texts, batch_size=64):
    pos = extract_pos_stats(doc)
    words = [token.text for token in doc if token.is_alpha]
    word_count = len(words)
    avg_word_length = np.mean([len(w) for w in words]) if words else 0
    char_count = len(doc.text)
    readability = flesch_reading_ease(doc.text)
    feats = [
        pos['noun_ratio'], pos['verb_ratio'], pos['adj_ratio'],
        pos['adv_ratio'], pos['pron_ratio'], pos['num_ratio'],
        char_count, word_count, avg_word_length, readability
    ]
    text_stats.append(feats)
text_stats = np.array(text_stats)

# TF-IDF
print("Fitting TF-IDF vectorizer...")
tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', analyzer='word')
tfidf_matrix = tfidf_vectorizer.fit_transform(texts)

# Combine features
X = hstack([tfidf_matrix, csr_matrix(text_stats)])
y = np.array(labels)

# Stratified Split
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, test_idx in sss.split(X, y):
    X_train, X_val = X[train_idx], X[test_idx]
    y_train, y_val = y[train_idx], y[test_idx]

# Train Calibrated Ensemble model
print("Training calibrated ensemble model...")
base_model = GradientBoostingClassifier(n_estimators=100)
clf = CalibratedClassifierCV(estimator=base_model, cv=3)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_val)
acc = accuracy_score(y_val, y_pred)
print(f"\n✅ Accuracy: {acc:.4f}")
print("\n🔲 Confusion Matrix:")
print(confusion_matrix(y_val, y_pred))

# Save artifacts
print("Saving model and data...")
joblib.dump(tfidf_vectorizer, "hc3_tfidf_vectorizer.pkl")
joblib.dump(clf, "hc3_model.pkl")
joblib.dump(X_val, "hc3_X_val.pkl")
joblib.dump(y_val, "hc3_y_val.pkl")
print("✅ Done.")

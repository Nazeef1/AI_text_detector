import streamlit as st
import joblib
import numpy as np
import spacy
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from textstat import flesch_reading_ease
import spacy.cli

spacy.cli.download("en_core_web_sm")

nlp = spacy.load("en_core_web_sm")
model = joblib.load("hc3_model.pkl")
tfidf = joblib.load("hc3_tfidf_vectorizer.pkl")

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

def predict_ai_text(text):
    doc = nlp(text)
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
    tfidf_feat = tfidf.transform([text])
    final_feat = hstack([tfidf_feat, csr_matrix([feats])])
    pred_proba = model.predict_proba(final_feat)[0]
    prediction = model.predict(final_feat)[0]
    return prediction, pred_proba


st.set_page_config(page_title="AI Text Detector", layout="centered")
st.title("🧠 AI Text Detector")
st.write("Enter any text below to find out if it's **AI-generated** or **Human-written**.")

user_input = st.text_area("🔍 Enter your text:", height=200)

if st.button("Analyze"):
    if user_input.strip():
        pred, prob = predict_ai_text(user_input)
        label = "🤖 AI-generated" if pred == 1 else "🧍 Human-written"
        st.markdown(f"### Prediction:")
        st.success(label)
        st.markdown(f"**Confidence (AI):** {prob[1]*100:.2f}%")
        st.markdown(f"**Confidence (Human):** {prob[0]*100:.2f}%")
    else:
        st.warning("Please enter some text to analyze.")
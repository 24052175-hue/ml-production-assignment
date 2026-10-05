"""
Streamlit front end for the sentiment model.

Run locally:
    streamlit run streamlit_app.py
"""
import os
import subprocess
import sys

import joblib
import streamlit as st

HERE = os.path.dirname(__file__)
MODEL_PATH = os.path.join(HERE, "model.pkl")


@st.cache_resource
def get_model():
    """Load model.pkl, training it first if it doesn't exist yet."""
    if not os.path.exists(MODEL_PATH):
        subprocess.run([sys.executable, os.path.join(HERE, "train.py")], check=True, cwd=HERE)
    return joblib.load(MODEL_PATH)


st.set_page_config(page_title="Sentiment Analyzer", page_icon="💬")

st.title("Sentiment Analyzer")
st.write(
    "Type a product review and the model predicts whether it's positive or negative. "
    "Trained on Amazon app reviews for the *Getting Started with ML in Production* workshop."
)

model = get_model()

text = st.text_area("Review text", placeholder="e.g. Not a good thing", height=120)

if st.button("Predict", type="primary"):
    if not text.strip():
        st.warning("Type some text first.")
    else:
        probs = model.predict_proba([text])[0]
        pred = int(probs.argmax())
        confidence = float(probs[pred])
        if pred == 1:
            st.success(f"**Positive** (confidence: {confidence:.2f})")
        else:
            st.error(f"**Negative** (confidence: {confidence:.2f})")
        st.progress(confidence, text=f"Model confidence: {confidence:.0%}")

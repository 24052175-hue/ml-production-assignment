"""
FastAPI app serving a sentiment analysis model.

Run locally:
    python train.py              # creates model.pkl
    uvicorn main:app --reload

Then open:
    http://127.0.0.1:8000/test   (web page to try the model)
    http://127.0.0.1:8000/docs   (interactive API docs)
"""
import os

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

app = FastAPI(
    title="Sentiment Analysis API",
    description="Predicts whether a product review is positive or negative.",
    version="2.0.0",
)

# Load the model once at startup
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    model = None


class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=1, description="The review text to analyze")


class PredictionResponse(BaseModel):
    sentiment: str
    confidence: float


@app.get("/")
def root():
    return {"message": "Sentiment API is running. Open /test to try it, or /docs for the API."}


@app.get("/health")
def health():
    """Basic health check endpoint — useful for deployment platforms."""
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run train.py first.")

    probs = model.predict_proba([request.text])[0]
    pred = int(probs.argmax())
    sentiment = "positive" if pred == 1 else "negative"
    return PredictionResponse(sentiment=sentiment, confidence=float(probs[pred]))


TEST_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Sentiment Tester</title>
  <style>
    body { font-family: Arial, sans-serif; max-width: 640px; margin: 40px auto; padding: 0 16px; }
    textarea { width: 100%; height: 90px; font-size: 16px; padding: 8px; box-sizing: border-box; }
    button { margin-top: 12px; padding: 10px 20px; font-size: 16px; cursor: pointer; }
    #result { margin-top: 20px; font-size: 18px; font-weight: bold; }
  </style>
</head>
<body>
  <h1>Sentiment Analyzer</h1>
  <textarea id="text" placeholder="Type a review, e.g. 'Not a good thing'"></textarea>
  <br>
  <button onclick="predict()">Predict</button>
  <div id="result"></div>

  <script>
    async function predict() {
      const text = document.getElementById("text").value.trim();
      const result = document.getElementById("result");
      if (!text) { result.textContent = "Type some text first."; return; }
      result.textContent = "Predicting...";
      try {
        const res = await fetch("/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: text })
        });
        const data = await res.json();
        if (!res.ok) { result.textContent = "Error: " + JSON.stringify(data.detail); return; }
        result.textContent = data.sentiment + " (confidence: " + data.confidence + ")";
      } catch (err) {
        result.textContent = "Could not reach the API: " + err;
      }
    }
  </script>
</body>
</html>
"""


@app.get("/test", response_class=HTMLResponse)
def test_page():
    """A simple web page for trying the model in the browser."""
    return TEST_PAGE

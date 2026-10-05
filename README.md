# ML in Production – Sentiment Analysis API

A sentiment model (TF-IDF + Logistic Regression) trained on Amazon app reviews,
served with **FastAPI**, packaged with **Docker**, with a **Streamlit** front end.
Built for the *Getting Started with ML in Production* workshop.

## 🌐 Live Demo

- **Test page:** https://ml-production-assignment.onrender.com/test
- **API docs:** https://ml-production-assignment.onrender.com/docs

> Hosted on Render's free tier — the first request may take ~50 seconds while the server wakes up.

## Project structure

```
.
├── datasets/amazon_dataset.csv   # 20,000 labelled reviews (1 = positive, 0 = negative)
├── train.py                      # trains the model and saves model.pkl
├── main.py                       # FastAPI app (/, /health, /predict, /test)
├── streamlit_app.py              # Streamlit web app
├── requirements.txt
├── Dockerfile
└── .dockerignore
```

## Run locally

```bash
python -m venv venv
venv\Scripts\activate          # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
python train.py                # creates model.pkl
uvicorn main:app --reload
```

Open http://127.0.0.1:8000/test to try it in the browser.

## Run with Docker

```bash
docker build -t sentiment-api .
docker run -p 8000:8000 sentiment-api
```

## Run the Streamlit app

```bash
streamlit run streamlit_app.py
```

## Example request

```bash
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "Not a good thing"}'
```

```json
{"sentiment": "negative", "confidence": 0.95}
```

## Endpoints

| Method | Path       | Description                          |
|--------|------------|--------------------------------------|
| GET    | `/`        | Welcome message                      |
| GET    | `/health`  | Health check (is the model loaded?)  |
| POST   | `/predict` | Predict sentiment of `{"text": ...}` |
| GET    | `/test`    | Web page to try the model            |

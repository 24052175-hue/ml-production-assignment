# ML in Production – Iris Prediction API

A scikit-learn model served with **FastAPI** and packaged with **Docker**.
Built for the *Getting Started with ML in Production* workshop.

## Project structure

```
.
├── train.py           # trains the model and saves model.pkl
├── main.py            # FastAPI app (/, /health, /predict)
├── requirements.txt   # Python dependencies
├── Dockerfile         # container image for the API
├── .dockerignore
└── datasets/          # sample dataset from the workshop
```

## Run locally (without Docker)

```bash
python -m venv venv
venv\Scripts\activate          # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
python train.py                # creates model.pkl
uvicorn main:app --reload
```

Open http://127.0.0.1:8000/docs to try the API.

## Run with Docker

```bash
docker build -t ml-api .
docker run -p 8000:8000 ml-api
```

Then open http://localhost:8000/docs.

## Example request

```bash
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
```

Response:

```json
{"prediction": 0, "class_name": "setosa"}
```

## Endpoints

| Method | Path       | Description                         |
|--------|------------|-------------------------------------|
| GET    | `/`        | Welcome message                     |
| GET    | `/health`  | Health check (is the model loaded?) |
| POST   | `/predict` | Predict the iris species            |

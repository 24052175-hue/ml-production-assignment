# Dockerfile for the ML prediction API (FastAPI + scikit-learn)
FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the training script and API code
COPY train.py main.py ./

# Train the model inside the image so model.pkl always exists
RUN python train.py

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

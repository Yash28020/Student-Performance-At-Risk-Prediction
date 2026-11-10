# Production Dockerfile for EduGuard AI Backend & Streamlit Service
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /tmp/matplotlib && chmod 777 /tmp/matplotlib

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY api ./api
COPY dashboard ./dashboard
COPY models ./models
COPY data ./data
COPY reports ./reports
COPY README.md LICENSE .env.example ./

RUN mkdir -p data/raw data/processed models reports/figures

EXPOSE 8000 8501

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

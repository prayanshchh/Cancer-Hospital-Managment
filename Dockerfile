FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements-docker.txt .
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements-docker.txt

COPY . .

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=5 \
  CMD python -c "from urllib.request import urlopen; urlopen('http://127.0.0.1:7860/healthz', timeout=5).read()"

CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--threads", "4", "--timeout", "600", "app:app"]

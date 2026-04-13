FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install -e . --no-deps

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

EXPOSE 8080 9090

CMD ["python", "dashboard/app.py", "--host", "0.0.0.0", "--port", "8080"]

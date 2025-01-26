FROM python:3.10-slim-buster AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels -r requirements.txt

FROM python:3.10-slim-buster

WORKDIR /app

# Copy wheels and requirements.txt from the builder stage
COPY --from=builder /app/wheels /wheels
COPY --from=builder /app/requirements.txt .

# Install Python dependencies from wheels
RUN pip install --no-cache /wheels/* \
    && python -m spacy download en_core_web_sm \
    && python -c "import nltk; nltk.download('punkt')"

# Copy the rest of the application code
COPY . .

# Create uploads directory
RUN mkdir -p uploads

# Expose ports
EXPOSE 8080 6333

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    FLASK_APP=app.py \
    QDRANT_HOST=qdrant \
    QDRANT_PORT=6333

# Define the command to run the application
CMD ["python", "app.py"]
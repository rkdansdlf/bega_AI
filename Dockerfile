# Use Python 3.11 as base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for building Python packages
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install dependencies with increased timeout and retries
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
    --default-timeout=1000 \
    --retries=5 \
    -r requirements.txt

# Copy source code
COPY . .

# Expose port 8001
EXPOSE 8001

# Run the FastAPI application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001", "--reload"]
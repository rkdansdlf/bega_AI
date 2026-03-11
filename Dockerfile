# Base image
FROM python:3.14-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/home/appuser/.local/bin:${PATH}"

WORKDIR /app

# Install runtime system dependencies needed by the app.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libpq5 \
    libjpeg62-turbo \
    libpng16-16 \
    zlib1g \
    curl \
    ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies with no pip cache to reduce disk usage.
COPY requirements.txt .
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --no-cache-dir --disable-pip-version-check -r requirements.txt && \
    rm -rf /root/.cache/pip

# Create non-root user
RUN useradd -m -u 1000 appuser

# Copy source code
COPY --chown=appuser:appuser . .

USER appuser

# Expose port 8001
EXPOSE 8001

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Run the FastAPI application (no --reload in production)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]

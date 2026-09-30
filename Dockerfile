# Base image
FROM python:3.14-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONNOUSERSITE=1 \
    PYTHONUNBUFFERED=1 \
    TZ=Asia/Seoul

WORKDIR /app

# Install runtime system dependencies needed by the app, including security updates.
RUN apt-get update && \
    apt-get upgrade -y && \
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

# Create the runtime identity and its only application-owned writable path.
RUN useradd -m -u 1000 appuser && \
    install -d -o appuser -g appuser -m 0750 /app/reports

# Keep startup code root-owned so the runtime identity cannot persist changes.
COPY app ./app
COPY scripts ./scripts
COPY docs ./docs
COPY evals ./evals
RUN find /app/app /app/scripts /app/docs /app/evals -type d -exec chmod 0555 {} + && \
    find /app/app /app/scripts /app/docs /app/evals -type f -perm /111 -exec chmod 0555 {} + && \
    find /app/app /app/scripts /app/docs /app/evals -type f ! -perm /111 -exec chmod 0444 {} +

USER appuser

# Expose port 8001
EXPOSE 8001

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8001/ready || exit 1

# Run the FastAPI application (no --reload in production; uvloop for async perf)
CMD ["/usr/local/bin/uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001", "--loop", "uvloop", "--workers", "1"]

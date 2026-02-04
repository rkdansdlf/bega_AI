# Stage 1: Build - Install dependencies and create wheels
FROM python:3.14-rc-slim AS builder

WORKDIR /build

# Install system dependencies for building Python packages
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Build wheels for all dependencies
RUN pip wheel --no-cache-dir -r requirements.txt -w /wheels

# Stage 2: Runtime - Minimal production image
FROM python:3.14-rc-slim

# Enable free-threading mode (Python 3.14+)


WORKDIR /app

# Install only runtime dependencies (libpq for psycopg3)
RUN apt-get update && apt-get install -y \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 appuser

# Copy wheels from builder and install
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/* && rm -rf /wheels

# Copy source code
COPY --chown=appuser:appuser . .

USER appuser

# Expose port 8001
EXPOSE 8001

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Run the FastAPI application (no --reload in production)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
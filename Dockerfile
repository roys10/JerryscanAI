# Multi-stage build to reduce final image size
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_NO_CACHE=1

WORKDIR /app

# System libs needed by image handling (no libgl1 needed for headless)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --no-cache-dir uv

# Copy dependency metadata for better layer caching
COPY pyproject.toml uv.lock* ./

# Install dependencies via `uv pip install --group default` so no `.venv` is created
RUN uv pip install --group default --extra-index-url https://download.pytorch.org/whl/cpu --target=/app/deps

# Runtime stage
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy installed dependencies from builder (target directory created by `uv pip install`)
COPY --from=builder /app/deps /usr/local/lib/python3.12/site-packages

# System libs needed by image handling
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy only the backend code
COPY backend/ ./backend/

EXPOSE 8000

CMD ["python", "backend/main.py"]

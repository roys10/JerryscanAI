# Multi-stage build to reduce final image size
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_NO_CACHE=1
ENV UV_NO_INSTALL_PROJECT=1

WORKDIR /app

# System libs needed by image handling (no libgl1 needed for headless)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --no-cache-dir uv

# Copy dependency metadata plus README (needed by hatch during build if export runs)
COPY pyproject.toml uv.lock* README.md ./

# Export a pinned requirements.txt and install via uv pip install into /app/deps (no .venv)
RUN uv export --no-dev --extra-index-url https://download.pytorch.org/whl/cpu --output-file requirements.txt && \
    uv pip install --requirements requirements.txt --target /app/deps

# Runtime stage
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy installed dependencies from builder (target directory created by `uv sync`)
COPY --from=builder /app/deps/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages

# System libs needed by image handling
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy only the backend code
COPY backend/ ./backend/

EXPOSE 8000

CMD ["python", "backend/main.py"]

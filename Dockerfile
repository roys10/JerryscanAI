FROM python:3.12-slim

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

# Install runtime deps into system Python
# Use the CPU-only PyTorch index to avoid 5GB+ of GPU binaries
RUN UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu \
    uv sync --no-dev --system

# Copy only the backend code
COPY backend/ ./backend/

EXPOSE 8000

CMD ["python", "backend/main.py"]
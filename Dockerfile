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

# Copy dependency metadata
COPY pyproject.toml uv.lock* ./

# Export locked runtime dependencies (excluding the root project),
# drop GPU-only packages, then install CPU-only torch backend.
RUN uv export --frozen --no-dev --no-emit-project --no-hashes --no-annotate --no-header --output-file requirements.txt && \
    grep -Ev '^(nvidia-|triton==|cuda-bindings==|cuda-pathfinder==)' requirements.txt > requirements.cpu.txt && \
    uv pip install --torch-backend cpu --requirements requirements.cpu.txt --target /app/deps

# Runtime stage
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy installed dependencies from builder (`uv pip --target` writes directly under /app/deps)
COPY --from=builder /app/deps /usr/local/lib/python3.12/site-packages

# System libs needed by image handling
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy only the backend code
COPY backend/ ./backend/

EXPOSE 8000

CMD ["python", "backend/main.py"]

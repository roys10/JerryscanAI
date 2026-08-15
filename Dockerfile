FROM python:3.12-slim AS builder

ENV UV_NO_CACHE=1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv

COPY pyproject.toml uv.lock* ./
RUN uv export --extra preprocess-rembg --no-dev --no-emit-project --no-hashes \
        --no-annotate --no-header --output-file requirements.txt \
    && grep -Ev '^(nvidia-|triton==|cuda-bindings==|cuda-pathfinder==)' requirements.txt \
        > requirements.cpu.txt \
    && uv pip install --torch-backend cpu --requirements requirements.cpu.txt \
        --target /app/deps

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app

COPY --from=builder /app/deps /usr/local/lib/python3.12/site-packages
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libxcb1 libgl1 libxext6 libsm6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY backend/ ./backend/
COPY training/ ./training/
COPY models/ ./models/

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]

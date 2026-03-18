FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_NO_CACHE=1

WORKDIR /app

# System libs needed by opencv/headless image handling
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --no-cache-dir uv

# Copy only dependency metadata first for better layer caching
COPY pyproject.toml ./

# Install runtime deps into system Python
# IMPORTANT:
# - this avoids creating .venv
# - this keeps the image smaller than uv sync
RUN uv pip install --system -r pyproject.toml \
    && rm -rf /root/.cache

# Copy only the backend code
COPY backend/ ./backend/

EXPOSE 8000

CMD ["python", "backend/main.py"]
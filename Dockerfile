FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency resolution
RUN pip install uv

# Set the working directory
WORKDIR /app

# Copy dependency files first
COPY pyproject.toml uv.lock ./

# Ensure backend directory is present, used in run setup
RUN mkdir -p backend/inference

# Sync dependencies using the CPU-only PyTorch index to save ~6GB of space
# Also disable the cache so downloaded wheels aren't saved in the final image
ENV UV_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cpu"
RUN uv sync --no-dev --no-cache

# Copy only the backend app
COPY backend/ backend/

# Expose the API port
EXPOSE 8000

# Start server using python directly
CMD ["python", "backend/main.py"]

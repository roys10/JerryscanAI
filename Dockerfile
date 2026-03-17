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

# Copy dependency files first (Deliberately omit uv.lock so it resolves fresh against the CPU index)
COPY pyproject.toml ./

# Ensure backend directory is present, used in run setup
RUN mkdir -p backend/inference

# Generate a fresh requirements list (ignoring lockfiles) and install it into the system Python
# We use the CPU-only PyTorch index to save ~6GB of space, and disable cache to keep the image small
RUN uv sync

COPY backend/ backend/

# Expose the API port
EXPOSE 8000

# Start server using python directly
CMD ["python", "backend/main.py"]

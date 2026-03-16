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

# Sync dependencies (system-wide in the container)
RUN uv sync --no-dev --system

# Copy the rest of the application
COPY . .

# Expose the API port
EXPOSE 8000

# Start server using python directly
CMD ["python", "backend/main.py"]

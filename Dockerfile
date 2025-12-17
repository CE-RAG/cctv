FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for PyTorch, CUDA and image processing
RUN apt-get update && apt-get install -y \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Note: This Dockerfile is for CPU-only deployment
# For GPU support, use Dockerfile.gpu instead

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/app/cache

# Copy dependency files first (for better Docker layer caching)
COPY pyproject.toml uv.lock* ./

# Sync dependencies using uv
RUN uv sync --frozen --no-dev

# Copy application code
COPY main.py .
# Copy images
COPY images ./images

# Expose the port the server runs on
EXPOSE 8000

# Create cache directory for model downloads
RUN mkdir -p /app/cache

# Run the application
CMD ["uv", "run", "main.py"]

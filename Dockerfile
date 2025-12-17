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

# Install CUDA libraries (CUDA 12.x compatible)
# This provides the CUDA runtime libraries needed by PyTorch
RUN apt-get update && apt-get install -y \
    cuda-cudart-12-4 \
    libcublas-12-4 \
    libcufft-12-4 \
    libcurand-12-4 \
    libcusolver-12-4 \
    libcusparse-12-4 \
    || apt-get install -y \
    cuda-cudart-12-1 \
    libcublas-12-1 \
    libcufft-12-1 \
    libcurand-12-1 \
    libcusolver-12-1 \
    libcusparse-12-1 \
    && rm -rf /var/lib/apt/lists/*

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

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/app/cache
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Create cache directory for model downloads
RUN mkdir -p /app/cache

# Run the application
CMD ["uv", "run", "main.py"]

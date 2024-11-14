# Use Python 3.9 as base
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install dependencies
RUN pip install rembg fastapi uvicorn requests aiohttp pillow python-multipart

# Create directories for temporary file storage
RUN mkdir -p /app/temp

# Create .u2net directory and download the model
RUN mkdir -p /root/.u2net && \
    wget https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx -O /root/.u2net/u2net.onnx

# Copy the server script
COPY server.py .

# Expose the port
EXPOSE 8000

# Run the server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
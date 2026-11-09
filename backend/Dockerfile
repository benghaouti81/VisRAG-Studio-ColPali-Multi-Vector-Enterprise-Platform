# Multi-Vector Visual Retrieval & VisRAG Backend Service
FROM python:3.11-slim

# System dependencies for PDF rendering (poppler, libmupdf, build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency definition
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Set environment defaults
ENV PYTHONUNBUFFERED=1 \
    DEVICE=cpu \
    PORT=8000

# Run FastAPI with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for document processing
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    libmagic1 \
    poppler-utils \
    tesseract-ocr \
    libtesseract-dev \
    libreoffice \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt', quiet=True)"

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Command will be overridden in docker-compose
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

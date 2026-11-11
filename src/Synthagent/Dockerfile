# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HOST=0.0.0.0
ENV PORT=8050

# Set work directory
WORKDIR /app

# Install system dependencies required for Kaleido (Plotly to image for PDF)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libnss3 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libxkbcommon0 \
    libgdk-pixbuf-2.0-0 \
    libgtk-3-0 \
    libasound2 \
    && rm -rf /var/lib/apt/lists/*


# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir gunicorn

# Copy project files
COPY . .

# Create cache directory to avoid volume mount race conditions
RUN mkdir -p /app/cache

# Expose port
EXPOSE 8050

# Run with Gunicorn for concurrent workers concurrency
CMD ["gunicorn", "-b", "0.0.0.0:8050", "--workers", "4", "--timeout", "120", "app:server"]

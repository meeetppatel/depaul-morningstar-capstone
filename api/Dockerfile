FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y \
    gcc g++ curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY main.py .
COPY artifacts/ ./artifacts/

# Expose port
EXPOSE 8080

# Run
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]

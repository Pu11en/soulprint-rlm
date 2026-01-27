FROM python:3.12-slim

WORKDIR /app

# Install git for pip install from GitHub
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir git+https://github.com/alexzhang13/rlm.git

# Copy app
COPY main.py .

# Expose port (Render uses PORT env var, default 10000)
EXPOSE 10000

# Run - use PORT env var from Render
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000}

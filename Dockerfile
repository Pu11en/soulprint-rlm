FROM python:3.12-slim

WORKDIR /app

# Install git (required for pip install from GitHub)
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# Copy and install requirements first (for caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy adapters and processors (ordered by change frequency for layer caching)
COPY adapters/ ./adapters/
COPY processors/ ./processors/

# Copy app
COPY main.py .

# Verify all modules import correctly at build time (fail fast)
RUN python -c "from adapters import download_conversations, update_user_profile, save_chunks_batch; print('Adapters OK')" && \
    python -c "from processors.conversation_chunker import chunk_conversations; from processors.fact_extractor import extract_facts_parallel; from processors.memory_generator import generate_memory_section; from processors.v2_regenerator import regenerate_sections_v2; from processors.full_pass import run_full_pass_pipeline; print('Processors OK')"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:${PORT:-10000}/health').raise_for_status()"

# Expose port (Render uses PORT env var)
EXPOSE 10000

# Run
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000}"]

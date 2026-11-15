# ==============================================================================
# Dockerfile: Marine Oil Spill Segmentation & Vessel Attribution Dashboard
# Suitable for Render Docker Runtime, AWS ECS, GCP Cloud Run, and Local Docker
# ==============================================================================
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8501 \
    MPLCONFIGDIR=/tmp/matplotlib_cache

WORKDIR /app

# Install lightweight system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Pre-generate synthetic radar and AIS benchmark files
RUN python generate_sample_data.py && \
    python -c "from src.setup_real_world_data import setup_all_real_world_data; setup_all_real_world_data()" || true

# Expose Streamlit default port
EXPOSE 8501

# Healthcheck for cloud container orchestration
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT}/_stcore/health || exit 1

# Launch Streamlit with dynamic cloud PORT binding
CMD ["sh", "-c", "streamlit run app.py --server.port ${PORT} --server.address 0.0.0.0 --server.headless true"]

FROM python:3.10.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Unity runtime deps + xvfb for headless display (safe default)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    bash \
    xvfb \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy python project
COPY training_manager/ /app/training_manager/
COPY config/ /app/config/

# Install python deps
RUN pip install --no-cache-dir -r /app/training_manager/requirements.txt

# Make imports like `from training_pipeline...` work
ENV PYTHONPATH=/app/training_manager

# Point your code to the Linux env inside the container
# (Only helps if your Paths() reads UNITY_ENV_PATH from env; if not, we’ll adjust Paths.)
ENV UNITY_ENV_PATH=/app/training_manager/builds/3DBall_Linux/3DBall.x86_64

# Ensure the Linux Unity binary is executable
RUN chmod +x /app/training_manager/builds/3DBall_Linux/3DBall.x86_64 || true

# Default command: help (override when running)
CMD ["python", "-m", "training_pipeline.cli.run_experiment", "-h"]

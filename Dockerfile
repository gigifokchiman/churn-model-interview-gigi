FROM python:3.10-slim

WORKDIR /app

# Define build arguments
ARG SRC_DIR=src
ARG ARTIFACTS_DIR=xgboost_20241113_035655

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# Copy source code and artifacts using the build arguments
COPY ${SRC_DIR}/ /app/src/
COPY ${ARTIFACTS_DIR}/ /app/artifacts/

# Set Python path
ENV PYTHONPATH=/app

# Default command
CMD ["python", "-m", "src.pipelines.inferencing_pipeline"]



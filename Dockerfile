FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

LABEL maintainer="Nicolas Carrion <ncarriong@estud.usfq.edu.ec>"
LABEL description="Diffusion-Based Academic Performance Prediction (PISA 2022)"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    unzip \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt /workspace/requirements.txt
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r /workspace/requirements.txt

# Copy source code and configuration
COPY configs/ /workspace/configs/
COPY src/ /workspace/src/
COPY run.py /workspace/run.py
COPY README.md /workspace/README.md

# Default entrypoint
CMD ["python", "run.py", "--step", "all"]

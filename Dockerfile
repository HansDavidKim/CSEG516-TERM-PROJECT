# Ubuntu 22.04.4 base image to match Google Colab's runtime
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Use bash with pipefail for predictable RUN behaviour
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Install common Colab system packages and enable the deadsnakes PPA for Python 3.12.11
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gnupg \
        software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        ffmpeg \
        git \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        unzip \
        wget \
        python3.12 \
        python3.12-dev \
        python3.12-venv && \
    rm -rf /var/lib/apt/lists/*

# Ensure pip is available and make Python 3.12 the default python/python3
RUN python3.12 -m ensurepip --upgrade && \
    ln -sf /usr/bin/python3.12 /usr/bin/python3 && \
    ln -sf /usr/bin/python3 /usr/local/bin/python && \
    python -m pip install --no-cache-dir --upgrade pip

WORKDIR /workspace

# Optionally install Python dependencies if requirements.txt exists and is non-empty
COPY requirements.txt ./
RUN if [[ -s requirements.txt ]]; then \
        python -m pip install --no-cache-dir -r requirements.txt; \
    fi

# Copy Python sources into the workspace image layer
COPY *.py ./

CMD ["python"]

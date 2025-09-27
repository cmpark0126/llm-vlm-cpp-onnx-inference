FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    git-lfs \
    python3 \
    python3-pip \
    python3.10-venv \
    wget \
    clang-format \
    clang-tidy \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies for baseline execution
COPY requirements.txt .
RUN python3 -m venv venv && \
    /venv/bin/python -m pip install --upgrade pip && \
    /venv/bin/pip install -r requirements.txt

WORKDIR /workspace

CMD ["/bin/bash"]

FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    git-lfs \
    python3 \
    python3-pip \
    wget \
    clang-format \
    clang-tidy \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies for baseline execution
COPY requirements.txt .
RUN pip3 install -r requirements.txt

WORKDIR /workspace

CMD ["/bin/bash"]

FROM --platform=linux/amd64 ubuntu:22.04

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

# Install ONNX Runtime - force x64 version
RUN wget https://github.com/microsoft/onnxruntime/releases/download/v1.19.0/onnxruntime-linux-x64-1.19.0.tgz \
    && tar -xzf onnxruntime-linux-x64-1.19.0.tgz \
    && cp -r onnxruntime-linux-x64-1.19.0/include/* /usr/local/include/ \
    && cp -r onnxruntime-linux-x64-1.19.0/lib/* /usr/local/lib/ \
    && rm -rf onnxruntime-linux-x64-1.19.0*

# Install Python dependencies for baseline execution
COPY requirements.txt .
RUN pip3 install -r requirements.txt

WORKDIR /workspace

CMD ["/bin/bash"]

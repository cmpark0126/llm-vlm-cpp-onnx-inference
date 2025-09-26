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
    htop \
    && rm -rf /var/lib/apt/lists/*

# Install ONNX Runtime - detect architecture and install appropriate version
RUN ARCH=$(uname -m) && \
    if [ "$ARCH" = "aarch64" ]; then \
        ONNX_ARCH="aarch64"; \
    elif [ "$ARCH" = "x86_64" ]; then \
        ONNX_ARCH="x64"; \
    else \
        echo "Unsupported architecture: $ARCH" && exit 1; \
    fi && \
    wget https://github.com/microsoft/onnxruntime/releases/download/v1.19.0/onnxruntime-linux-${ONNX_ARCH}-1.19.0.tgz \
    && tar -xzf onnxruntime-linux-${ONNX_ARCH}-1.19.0.tgz \
    && cp -r onnxruntime-linux-${ONNX_ARCH}-1.19.0/include/* /usr/local/include/ \
    && cp -r onnxruntime-linux-${ONNX_ARCH}-1.19.0/lib/* /usr/local/lib/ \
    && rm -rf onnxruntime-linux-${ONNX_ARCH}-1.19.0*

# Install Python dependencies for baseline execution
COPY requirements.txt .
RUN pip3 install -r requirements.txt

WORKDIR /workspace

CMD ["/bin/bash"]

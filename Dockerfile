FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

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
    libopencv-dev \
    python3-opencv \
    && rm -rf /var/lib/apt/lists/*

# Install ONNX Runtime
RUN ARCH=$(uname -m) && \
    if [ "$ARCH" = "aarch64" ]; then \
        ONNX_ARCH="aarch64"; \
    elif [ "$ARCH" = "x86_64" ]; then \
        ONNX_ARCH="x64"; \
    else \
        echo "Unsupported architecture: $ARCH" && exit 1; \
    fi && \
    ONNX_VERSION="1.19.0" && \
    ONNX_FILE="onnxruntime-linux-${ONNX_ARCH}-${ONNX_VERSION}.tgz" && \
    ONNX_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VERSION}/${ONNX_FILE}" && \
    wget "$ONNX_URL" && \
    tar -xzf "$ONNX_FILE" && \
    cp -r "onnxruntime-linux-${ONNX_ARCH}-${ONNX_VERSION}/include/"* /usr/include/ && \
    cp -r "onnxruntime-linux-${ONNX_ARCH}-${ONNX_VERSION}/lib/"* /usr/lib/ && \
    rm -rf "$ONNX_FILE" "onnxruntime-linux-${ONNX_ARCH}-${ONNX_VERSION}"

# Install Python dependencies for baseline execution
COPY requirements.txt .
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install -r requirements.txt

WORKDIR /workspace

CMD ["/bin/bash"]

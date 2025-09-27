#!/bin/bash

# Install ONNX Runtime - detect architecture and install appropriate version

echo "Detecting system architecture..."
ARCH=$(uname -m)

if [ "$ARCH" = "aarch64" ]; then
    ONNX_ARCH="aarch64"
elif [ "$ARCH" = "x86_64" ]; then
    ONNX_ARCH="x64"
else
    echo "Unsupported architecture: $ARCH"
    exit 1
fi

echo "Architecture: $ARCH -> ONNX Runtime: $ONNX_ARCH"

# Download and extract ONNX Runtime
ONNX_VERSION="1.19.0"
ONNX_FILE="onnxruntime-linux-${ONNX_ARCH}-${ONNX_VERSION}.tgz"
ONNX_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VERSION}/${ONNX_FILE}"

echo "Downloading ONNX Runtime from: $ONNX_URL"
wget "$ONNX_URL"

if [ $? -eq 0 ]; then
    echo "Download successful. Extracting..."
    tar -xzf "$ONNX_FILE"

    if [ $? -eq 0 ]; then
        echo "Extraction successful!"
        echo "ONNX Runtime installed in: onnxruntime-linux-${ONNX_ARCH}-${ONNX_VERSION}/"

        # Clean up downloaded file
        rm "$ONNX_FILE"
        echo "Cleaned up downloaded archive."
    else
        echo "Extraction failed!"
        exit 1
    fi
else
    echo "Download failed!"
    exit 1
fi

echo "ONNX Runtime installation completed successfully!"
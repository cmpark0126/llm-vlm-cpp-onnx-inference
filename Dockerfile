FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    python3 \
    python3-pip \
    wget \
    clang-format \
    clang-tidy \
    && rm -rf /var/lib/apt/lists/*



WORKDIR /workspace

CMD ["/bin/bash"]

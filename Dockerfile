# Wyoming Faster Whisper — CUDA build for NVIDIA Jetson Orin Nano
# Host: JetPack 6.x, CUDA 12.6, driver 540.x
# Base image provides cuBLAS + cuDNN 9 runtime libraries (arm64 supported).
# Run with: docker run --runtime nvidia ...

FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

WORKDIR /usr/src

RUN \
    apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN \
    python3 -m venv .venv \
    && .venv/bin/pip3 install --no-cache-dir -U \
        setuptools \
        wheel \
    && .venv/bin/pip3 install --no-cache-dir \
        wyoming-faster-whisper==3.3.1

WORKDIR /
COPY run.sh ./

EXPOSE 10300

ENTRYPOINT ["bash", "/run.sh"]

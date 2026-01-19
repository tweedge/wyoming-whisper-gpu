FROM nvidia/cuda:12.3.2-base-ubuntu22.04

WORKDIR /usr/src

RUN \
    apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libavdevice-dev \
        libavfilter-dev \
        libavformat-dev \
        libswscale-dev \
        pkg-config \
        python3 \
        python3-dev \
        python3-pip \
        nvidia-cuda-toolkit

RUN \
    pip3 install --upgrade --no-cache-dir pip \
    && pip3 install --no-cache-dir -U \
        setuptools \
        wheel \
    && pip3 install --no-cache-dir torch \
    && pip3 install --no-cache-dir wyoming-faster-whisper==3.0.2

WORKDIR /
COPY run.sh ./

EXPOSE 10300

ENTRYPOINT ["bash", "/run.sh"]

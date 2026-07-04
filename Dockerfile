# Wyoming Faster Whisper — CUDA build for NVIDIA Jetson Orin Nano (SM87)
# JetPack 6.x / L4T r36.4, CUDA 12.6, driver 540.x
#
# The PyPI ctranslate2 wheel for aarch64 is CPU-only, so this Dockerfile
# compiles ctranslate2 from source with CUDA + cuDNN + OpenBLAS support
# using a multi-stage build.
#
# Stage 1 (builder): L4T CUDA devel image — provides nvcc and CUDA headers
# Stage 2 (runtime): L4T CUDA runtime image — smaller final image
#
# Build: docker build -t wyoming-whisper-gpu .
# Run:   docker run --runtime nvidia --network host \
#            -v $HOME/.cache/faster-whisper:/data \
#            -e MODEL=tiny -e LANGUAGE=en -e COMPUTE_TYPE=int8 \
#            wyoming-whisper-gpu

# CUDA SM architecture: 87 = Orin Nano / Orin NX / AGX Orin
#                       72 = Xavier NX / AGX Xavier
ARG CUDA_ARCH=87

ARG CTRANSLATE2_VERSION=4.5.0
ARG WYOMING_VERSION=3.3.1

# =============================================================================
# STAGE 1: Builder — compile ctranslate2 with CUDA support
# =============================================================================
FROM nvcr.io/nvidia/12.6.11-devel:12.6.11-devel-aarch64-ubuntu22.04 AS builder

ARG CUDA_ARCH
ARG CTRANSLATE2_VERSION

# Add Jetson apt repo (provides cuDNN packages not bundled in L4T images)
RUN echo "deb https://repo.download.nvidia.com/jetson/common r36.4 main" \
        >> /etc/apt/sources.list.d/nvidia-l4t-apt-source.list \
    && echo "deb https://repo.download.nvidia.com/jetson/t234 r36.4 main" \
        >> /etc/apt/sources.list.d/nvidia-l4t-apt-source.list \
    && apt-key adv --fetch-keys https://repo.download.nvidia.com/jetson/jetson-ota-public.asc

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        pkg-config \
        python3 \
        python3-pip \
        python3-venv \
        python3-dev \
        pybind11-dev \
        # OpenBLAS replaces Intel MKL (MKL is x86-only)
        libopenblas-dev \
        # libgomp1 provides libgomp.so.1 (GNU OpenMP, used by -DOPENMP_RUNTIME=COMP)
        libgomp1 \
        # cuDNN for CUDA 12 (required by CTranslate2's cuDNN backend)
        libcudnn9-dev-cuda-12 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Build CTranslate2 C++ library from source with CUDA + cuDNN + OpenBLAS
RUN git clone --recursive --depth 1 \
        --branch v${CTRANSLATE2_VERSION} \
        https://github.com/OpenNMT/CTranslate2.git \
    && cmake -S CTranslate2 -B CTranslate2/build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/opt/ctranslate2 \
        -DWITH_CUDA=ON \
        -DWITH_CUDNN=ON \
        -DWITH_MKL=OFF \
        -DWITH_OPENBLAS=ON \
        -DOPENMP_RUNTIME=COMP \
        -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
        -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} \
    && cmake --build CTranslate2/build --parallel $(nproc) \
    && cmake --install CTranslate2/build

# Build the CTranslate2 Python wheel against the library we just compiled
RUN python3 -m venv /opt/ct2venv \
    && /opt/ct2venv/bin/pip install --no-cache-dir \
        setuptools wheel pybind11 \
    && CT2_CUDA_ARCHS=${CUDA_ARCH} \
       CT2_CUDA_ROOT=/usr/local/cuda \
       CTRANSLATE2_ROOT=/opt/ctranslate2 \
       CMAKE_ARGS="-DWITH_CUDA=ON -DWITH_CUDNN=ON -DWITH_MKL=OFF -DWITH_OPENBLAS=ON \
                   -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
                   -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH}" \
       CFLAGS="-I/opt/ctranslate2/include" \
       LDFLAGS="-L/opt/ctranslate2/lib" \
       /opt/ct2venv/bin/pip wheel \
           --no-build-isolation \
           --no-cache-dir \
           -w /build/wheels \
           /build/CTranslate2/python

# =============================================================================
# STAGE 2: Runtime — install wyoming-faster-whisper with CUDA ctranslate2
# =============================================================================
FROM nvcr.io/nvidia/l4t-cuda:12.6.11-runtime

ARG CUDA_ARCH
ARG WYOMING_VERSION

# Add Jetson apt repo for cuDNN runtime library
RUN echo "deb https://repo.download.nvidia.com/jetson/common r36.4 main" \
        >> /etc/apt/sources.list.d/nvidia-l4t-apt-source.list \
    && echo "deb https://repo.download.nvidia.com/jetson/t234 r36.4 main" \
        >> /etc/apt/sources.list.d/nvidia-l4t-apt-source.list \
    && apt-key adv --fetch-keys https://repo.download.nvidia.com/jetson/jetson-ota-public.asc

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        # OpenBLAS runtime (required by ctranslate2 built with OpenBLAS)
        libopenblas0 \
        # libgomp1 provides libgomp.so.1 (GNU OpenMP, linked by ctranslate2 via -DOPENMP_RUNTIME=COMP)
        libgomp1 \
        # cuDNN runtime for CUDA 12
        libcudnn9-cuda-12 \
    && rm -rf /var/lib/apt/lists/*

# Copy the compiled CTranslate2 C++ library and Python wheel from builder
COPY --from=builder /opt/ctranslate2 /opt/ctranslate2
COPY --from=builder /build/wheels /tmp/wheels

WORKDIR /usr/src

# Install wyoming-faster-whisper, then replace the CPU-only ctranslate2
# that pip pulls in as a dependency with our CUDA-enabled build
RUN python3 -m venv .venv \
    && .venv/bin/pip3 install --no-cache-dir -U \
        setuptools \
        wheel \
    && .venv/bin/pip3 install --no-cache-dir \
        wyoming-faster-whisper==${WYOMING_VERSION} \
    && .venv/bin/pip3 install --no-cache-dir --force-reinstall \
        /tmp/wheels/ctranslate2-*.whl \
    && rm -rf /tmp/wheels

# Make the CTranslate2 C++ shared library visible at runtime
ENV LD_LIBRARY_PATH="/opt/ctranslate2/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}"

WORKDIR /
COPY run.sh warmup.py ./

EXPOSE 10300

ENTRYPOINT ["bash", "/run.sh"]

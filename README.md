# wyoming-whisper-gpu

A CUDA-enabled Docker image for [wyoming-faster-whisper](https://github.com/rhasspy/wyoming-faster-whisper), targeting the **NVIDIA Jetson Orin Nano** (JetPack 6.x, CUDA 12.6).

GPU acceleration is provided by [faster-whisper](https://github.com/SYSTRAN/faster-whisper) via [CTranslate2](https://github.com/OpenNMT/CTranslate2/), which uses cuBLAS and cuDNN directly — no PyTorch required at runtime.

## Requirements

- NVIDIA Jetson Orin Nano running JetPack 6.x (CUDA 12.6, driver 540.x)
- [NVIDIA Container Runtime](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed on the host (`nvidia-container-runtime`)
- Docker

## Build

```bash
docker build -t wyoming-whisper-gpu .
```

The image is based on `nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04` (arm64 compatible), which provides the cuBLAS and cuDNN 9 libraries required by CTranslate2 for GPU inference.

## Run

```bash
sudo docker run -d \
    --runtime=nvidia \
    --network host \
    -v $HOME/.cache/faster-whisper:/data \
    -e MODEL='tiny' \
    -e LANGUAGE='en' \
    -e COMPUTE_TYPE='int8' \
    -e BEAM_SIZE=5 \
    --name wyoming-whisper-tiny-en \
    wyoming-whisper-gpu
```

`--network host` exposes port 10300 directly on the host — convenient for Home Assistant on the same machine or LAN. Models are downloaded to `~/.cache/faster-whisper` on first run.

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `MODEL` | `tiny-int8` | Whisper model name (see below) |
| `LANGUAGE` | _(auto-detect)_ | Language code, e.g. `en`, `de`, `fr` |
| `COMPUTE_TYPE` | `default` | CTranslate2 compute type: `int8`, `float16`, `default` |
| `BEAM_SIZE` | `0` (auto) | Beam search width; auto selects 1 on ARM, 5 on x86 |

Available models: `tiny-int8`, `tiny`, `base-int8`, `base`, `small-int8`, `small`, `medium-int8`, `medium`, `large-v1`, `large-v2`, `large-v3`, and any HuggingFace model ID (e.g. `Systran/faster-distil-whisper-small.en`).

Additional CLI flags can be appended after the image name and will be passed through to `wyoming_faster_whisper` directly.

## Home Assistant / Wyoming Protocol

This image exposes a [Wyoming protocol](https://github.com/rhasspy/wyoming) STT endpoint on port **10300**, compatible with the Home Assistant Wyoming integration.

## Notes

- The NVIDIA Container Runtime passes the host CUDA driver into the container; the image does not bundle its own driver.
- cuBLAS and cuDNN 9 come from the `nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04` base image.
- `--device cuda` and `--compute-type int8` (implied by `tiny-int8`) are suitable for the Orin Nano's integrated GPU memory constraints.

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

**The build takes 20–40 minutes** on a Jetson Orin Nano — it compiles [CTranslate2](https://github.com/OpenNMT/CTranslate2) from source with CUDA support. This is unavoidable: the PyPI `aarch64` wheels for `ctranslate2` are CPU-only and will produce `ValueError: This CTranslate2 package was not compiled with CUDA support` at runtime.

The build uses two stages:
- **Stage 1** (`nvcr.io/nvidia/12.6.11-devel`): compiles CTranslate2 v4.5.0 from source against CUDA 12.6, cuDNN 9, and OpenBLAS (SM87 / Orin architecture). Produces a CUDA-enabled Python wheel.
- **Stage 2** (`nvcr.io/nvidia/l4t-cuda:12.6.11-runtime`): installs `wyoming-faster-whisper`, then replaces the CPU-only `ctranslate2` pip dependency with the wheel from Stage 1.

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


## Startup and warmup

On first run (or when the model isn't cached), the container downloads the model from HuggingFace before the server starts. On subsequent starts the model loads from `/data`.

After the server is ready, `run.sh` automatically runs two real transcription requests using 1 second of synthetic silence. This forces CUDA kernel JIT compilation and ensures the model is fully resident in GPU memory before the first real client request arrives. Without warmup, the first real request bears the full JIT cost; with it, that cost is paid at startup instead.

Example startup log:

```
[run.sh] Waiting for Wyoming server to be ready on port 10300...
INFO:httpx:HTTP Request: GET https://huggingface.co/.../faster-whisper-tiny/revision/main "HTTP/1.1 200 OK"
INFO:__main__:Ready
[run.sh] Server is ready. Running warmup...
[warmup] Connecting to Wyoming server at 127.0.0.1:10300
INFO:faster_whisper:Processing audio with duration 00:01.000
INFO:faster_whisper:Detected language 'en' with probability 0.29
[warmup] Run 1/2: 18268 ms  transcript=(empty — expected for silence)
INFO:faster_whisper:Processing audio with duration 00:01.000
INFO:faster_whisper:Detected language 'en' with probability 0.29
[warmup] Run 2/2: 491 ms  transcript=(empty — expected for silence)
[warmup] Done — model is warm and ready.
[run.sh] Warmup complete. Wyoming faster-whisper is ready for requests.
```

Run 1 is slow (18s here) due to CUDA JIT compilation. Run 2 shows the much lower steady-state latency that subsequent real requests will see.

## Notes

- The NVIDIA Container Runtime passes the host CUDA driver into the container; the image does not bundle its own driver.
- cuDNN 9 is installed from NVIDIA's Jetson apt repo (`repo.download.nvidia.com/jetson`); it is not bundled in the L4T base images.
- CTranslate2 is compiled for **SM87** (Orin Nano / Orin NX / AGX Orin). For Xavier (SM72), rebuild with `--build-arg CUDA_ARCH=72`.
- `--device cuda` and `--compute-type int8` are appropriate for the Orin Nano's shared GPU memory constraints.
- Standard `nvidia/cuda` Docker Hub images are **x86-only** for CUDA support; Jetson requires L4T (`nvcr.io/nvidia/l4t-*`) images.

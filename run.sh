#!/usr/bin/env bash
set -euo pipefail
cd /usr/src

# Environment variable overrides (used by docker run -e ...)
# MODEL: whisper model name, default tiny-int8
# LANGUAGE: language code (e.g. en), default auto-detect
# COMPUTE_TYPE: e.g. int8, float16, default (let faster-whisper choose)
# BEAM_SIZE: beam search width, 0 = auto (ARM auto-selects 1)

MODEL="${MODEL:-tiny-int8}"
LANGUAGE="${LANGUAGE:-}"
COMPUTE_TYPE="${COMPUTE_TYPE:-default}"
BEAM_SIZE="${BEAM_SIZE:-0}"

ARGS=(
    --model "${MODEL}"
    --device cuda
    --compute-type "${COMPUTE_TYPE}"
    --beam-size "${BEAM_SIZE}"
    --uri 'tcp://0.0.0.0:10300'
    --data-dir /data
    --download-dir /data
)

if [ -n "${LANGUAGE}" ]; then
    ARGS+=(--language "${LANGUAGE}")
fi

# Start Wyoming server in the background
.venv/bin/python3 -m wyoming_faster_whisper "${ARGS[@]}" "$@" &
SERVER_PID=$!

# Forward signals to the server process so Docker stop works cleanly
trap 'kill -TERM "$SERVER_PID"; wait "$SERVER_PID"' TERM INT

# Wait for the server to accept TCP connections (timeout: 120s)
echo "[run.sh] Waiting for Wyoming server to be ready on port 10300..."
TIMEOUT=120
ELAPSED=0
while ! (echo > /dev/tcp/127.0.0.1/10300) 2>/dev/null; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[run.sh] ERROR: Server process exited before becoming ready." >&2
        exit 1
    fi
    if [ "$ELAPSED" -ge "$TIMEOUT" ]; then
        echo "[run.sh] ERROR: Server did not become ready within ${TIMEOUT}s." >&2
        exit 1
    fi
    sleep 1
    ELAPSED=$((ELAPSED + 1))
done
echo "[run.sh] Server is ready. Running warmup..."

# Run two real transcription requests to warm CUDA kernels and ensure the
# model is fully resident in GPU memory before real traffic arrives.
.venv/bin/python3 /warmup.py --host 127.0.0.1 --port 10300 --runs 2

echo "[run.sh] Warmup complete. Wyoming faster-whisper is ready for requests."

# Wait for the server to exit (keeps the container alive)
wait "$SERVER_PID"

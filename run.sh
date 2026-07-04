#!/usr/bin/env bash
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

.venv/bin/python3 -m wyoming_faster_whisper "${ARGS[@]}" "$@"

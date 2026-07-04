#!/usr/bin/env bash
cd /usr/src
.venv/bin/python3 -m wyoming_faster_whisper \
    --model tiny-int8 \
    --device cuda \
    --uri 'tcp://0.0.0.0:10300' \
    --data-dir /data \
    --download-dir /data "$@"

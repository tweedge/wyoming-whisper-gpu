Well, whoops. This `wyoming-whisper-gpu` repository may not actually put `wyoming-faster-whisper` on the GPU. It appears to run on CPU - note no calls to `cuda()` in Torch: https://github.com/rhasspy/wyoming-faster-whisper/blob/main/wyoming_faster_whisper/transformers_whisper.py

It is a good Dockerfile and does boot a Wyoming STT endpoint which uses `faster-whisper` on the backend, but just be mindful that `faster-whisper` is CPU-only as written AFAIK.

I might eventually do up a copy of `wyoming-faster-whisper` with GPU acceleration, but we'll see.

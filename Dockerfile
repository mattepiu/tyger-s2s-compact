# tyger v1 — one GPU, up to 4 conversations, one venv (MOUTH: Gemma 4 E4B + VoxCPM2 on vLLM/vLLM-Omni; EARS:
# nemotron + endpoint head). See README.md and DEVELOPING.md.
#
# torch cu130 wheels bundle the CUDA 13 runtime, so a plain python base works on a host with an sm_120
# (Blackwell / RTX 5090) GPU and a CUDA-13-capable driver (>=590). No weights are baked in: models download
# on first start into the mounted HF cache.
#
# Build:  docker build -t tyger-v1 .
# Run:    docker run --gpus all --rm -p 7860:7860 \
#           --env-file .env \
#           -e HF_HOME=/hf -v /workspace/.hf_home:/hf \
#           -v "$PWD/voices":/app/voices:ro \
#           tyger-v1
#         then SSH-forward 7860 and open http://localhost:7860  (see README for HTTPS).
# API mode, for another app (README -> "API mode"): no pages (none are built in this image anyway), its origin, a
# conversation's length and the token's secret, all as env vars:
#           -e TYGER_PAGES=false -e TYGER_ALLOWED_ORIGINS=https://app.example.com \
#           -e TYGER_MAX_SESSION_S=180 -e TYGER_API_SECRET=...
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 git ca-certificates build-essential && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv

WORKDIR /app
COPY requirements.txt ./
RUN uv venv /app/.venv --python 3.12 && \
    uv pip install --python /app/.venv/bin/python --no-cache -r requirements.txt --torch-backend=cu130

COPY app ./app
COPY static ./static
COPY config ./config
COPY run.sh stop.sh ./
RUN chmod +x run.sh stop.sh

ENV HF_HOME=/hf HF_HUB_DISABLE_XET=1
EXPOSE 7860
ENTRYPOINT ["./run.sh"]

# tyger v1 — one GPU, up to 4 conversations, one venv (MOUTH: Gemma 4 E4B + VoxCPM2 on vLLM/vLLM-Omni; EARS:
# nemotron + endpoint head). How to build, run and check it: README.md -> "Docker" (and "RunPod + Cloudflare").
#
# torch cu130 wheels bundle the CUDA 13 runtime, so a plain python base works on a host with an sm_120
# (Blackwell / RTX 5090) GPU, a CUDA-13-capable driver (>=590) and the NVIDIA Container Toolkit. No weights are baked
# in: the models download on the first start into the /hf volume. The voice clips ARE baked in (voices/ of the build
# machine; the user's call until voices come from a database): they're personal data, so push the image to a private
# registry only. .dockerignore keeps the build context to what's copied below: never .env or .venv.
#
# Build:  docker build -t tyger-v1 .          (voices/ must hold the clips: mira.wav, aria.wav, kai.wav, rex.wav)
# Run:    docker run -d --name tyger --restart unless-stopped --gpus all --ipc=host -p 7860:7860 \
#           --env-file .env -v tyger-hf:/hf -v tyger-cache:/root/.cache -v tyger-triton:/root/.triton \
#           tyger-v1
#         .env: HF_TOKEN, OPENAI_API_KEY (the default config's smart model), and for API mode TYGER_API_SECRET
#         and TYGER_PAGES=false. --ipc=host: vLLM's processes share memory. The volumes: the models and the
#         compiled kernels, kept across containers. --restart: a slow first download can outlast run.sh's wait.
#         CLOUDFLARE_TUNNEL_TOKEN: run.sh also starts a Cloudflare Tunnel to :7860 (README -> "RunPod + Cloudflare").
# Check:  docker logs -f tyger (until "EARS ready"), then docker exec tyger .venv/bin/python scripts/stack/smoke.py
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 git ca-certificates build-essential && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv

# cloudflared, for the tunnel run.sh starts when CLOUDFLARE_TUNNEL_TOKEN is set. Pinned, and checked against the
# SHA-256 in the release notes (github.com/cloudflare/cloudflared/releases). No curl in the slim image: python fetches.
ARG CLOUDFLARED_VERSION=2026.9.3
ARG CLOUDFLARED_SHA256=77e26d8d900e0b8469f416239d14b5f296525fdf79fee6f511ef55609e3fbac2
RUN python3 -c "import sys, urllib.request; urllib.request.urlretrieve(sys.argv[1], '/usr/local/bin/cloudflared')" \
        "https://github.com/cloudflare/cloudflared/releases/download/${CLOUDFLARED_VERSION}/cloudflared-linux-amd64" && \
    echo "${CLOUDFLARED_SHA256}  /usr/local/bin/cloudflared" | sha256sum -c - && \
    chmod +x /usr/local/bin/cloudflared && cloudflared --version

WORKDIR /app
COPY requirements.txt ./
RUN uv venv /app/.venv --python 3.12 && \
    uv pip install --python /app/.venv/bin/python --no-cache -r requirements.txt --torch-backend=cu130

COPY app ./app
COPY static ./static
COPY config ./config
COPY scripts/stack ./scripts/stack
COPY voices ./voices
COPY run.sh stop.sh ./
RUN chmod +x run.sh stop.sh

ENV HF_HOME=/hf HF_HUB_DISABLE_XET=1
EXPOSE 7860
ENTRYPOINT ["./run.sh"]

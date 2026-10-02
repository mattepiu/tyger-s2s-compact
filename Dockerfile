# tyger v1 — one GPU, one conversation. Two venvs (EARS tf5.17 / MOUTH tf4.57) because nemotron needs
# transformers>=5.13 while MOSS-TTS-Nano's remote code runs on 4.57 (see NOTES.md).
#
# torch cu130 wheels bundle the CUDA 13 runtime, so a plain python base works on a host with an sm_120
# (Blackwell / RTX 5090) GPU and a CUDA-13-capable driver (>=590). No weights are baked in: models download
# on first start into the mounted HF cache.
#
# Build:  docker build -t tyger-v1 v1/
# Run:    docker run --gpus all --rm -p 7860:7860 \
#           --env-file v1/.env \
#           -e HF_HOME=/hf -v /workspace/.hf_home:/hf \
#           -v /workspace/voices:/voices \
#           -v /workspace/tyger/ref.wav:/voices/ref.wav:ro \
#           tyger-v1
#         then SSH-forward 7860 and open http://localhost:7860  (see README for HTTPS).
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 git ca-certificates && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-ears.txt requirements-mouth.txt ./

# two isolated venvs
RUN python -m venv /opt/venv-ears  && /opt/venv-ears/bin/pip  install --no-cache-dir -U pip wheel && \
    /opt/venv-ears/bin/pip  install --no-cache-dir -r requirements-ears.txt && \
    python -m venv /opt/venv-mouth && /opt/venv-mouth/bin/pip install --no-cache-dir -U pip wheel && \
    /opt/venv-mouth/bin/pip install --no-cache-dir -r requirements-mouth.txt

COPY app ./app
COPY static ./static
COPY config.yaml run.sh ./
RUN chmod +x run.sh

ENV EARS_VENV=/opt/venv-ears MOUTH_VENV=/opt/venv-mouth \
    HF_HOME=/hf HF_HUB_DISABLE_XET=1 PORT=7860
# default voice dir + moss snapshot dir point at mount-friendly paths; override in config.yaml if needed
EXPOSE 7860
ENTRYPOINT ["./run.sh"]

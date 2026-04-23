# -----------------------------------------------------------------------------
# Base image — Python 3.12 slim keeps the image small while matching the
# version used in CI and the local virtual environment.
# -----------------------------------------------------------------------------
FROM python:3.12-slim

# -----------------------------------------------------------------------------
# Environment — prevents .pyc files being written to disk and ensures all
# Python output is flushed immediately (important for Docker log streaming).
# -----------------------------------------------------------------------------
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# -----------------------------------------------------------------------------
# Working directory — all subsequent commands run relative to /app.
# -----------------------------------------------------------------------------
WORKDIR /app

# -----------------------------------------------------------------------------
# System dependencies:
#   ffmpeg        — audio and video decoding (librosa, moviepy, openai-whisper)
#   libsm6        — OpenCV runtime requirement (shared memory)
#   libxext6      — OpenCV runtime requirement (X11 extensions)
#   libglib2.0-0  — GLib runtime used by several CV and audio libraries
#   git           — required by DVC for SCM integration
# -----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsm6 \
        libxext6 \
        libglib2.0-0 \
        git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# Python dependencies — copied before the rest of the source so Docker can
# cache this layer and skip re-installation when only source code changes.
# -----------------------------------------------------------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# -----------------------------------------------------------------------------
# spaCy language model — downloaded after pip install so it benefits from the
# cached dependency layer above.
# -----------------------------------------------------------------------------
RUN python -m spacy download en_core_web_sm

# -----------------------------------------------------------------------------
# Application source — copied last to maximise layer cache reuse during builds.
# -----------------------------------------------------------------------------
COPY . .

# -----------------------------------------------------------------------------
# Runtime directories — created here so the container always has them even
# when host volumes are not mounted (e.g. in CI or standalone runs).
# -----------------------------------------------------------------------------
RUN mkdir -p data/raw data/extracted reports logs

# -----------------------------------------------------------------------------
# Port — Flask API listens on 5000 by default (matches dq_config.yaml).
# -----------------------------------------------------------------------------
EXPOSE 5000

# -----------------------------------------------------------------------------
# Entrypoint — starts the Flask API; host 0.0.0.0 is required so the port
# is reachable from outside the container.
# -----------------------------------------------------------------------------
CMD ["python", "src/api/app.py"]

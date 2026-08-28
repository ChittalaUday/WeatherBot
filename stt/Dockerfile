FROM python:3.11-slim

# Weights (~2.4 GB) are pulled on first start into the hf-cache volume, not baked into
# the image.
ENV HF_HOME=/cache MODEL_PATH=nvidia/nemotron-3.5-asr-streaming-0.6b

RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 ffmpeg \
 && rm -rf /var/lib/apt/lists/*

# PyPI torch drags in 3.5GB of CUDA wheels on every arch, dead weight without a GPU.
# Build with --build-arg TORCH_INDEX=https://pypi.org/simple to get the CUDA build back.
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir --index-url "${TORCH_INDEX}" --extra-index-url https://pypi.org/simple torch \
 && pip install --no-cache-dir "transformers>=5.13" soundfile librosa websockets==13.1

COPY server.py test_stream.py /app/
RUN useradd -m app && mkdir -p /cache && chown -R app /app /cache
USER app
EXPOSE 2700
CMD ["python", "/app/server.py"]

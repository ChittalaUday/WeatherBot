#!/usr/bin/env bash
# Recreate the two gitignored inputs of this folder: the NeMo-Speech.cpp checkout
# and the ASR weights. Idempotent; run before `docker compose build`.
set -euo pipefail
cd "$(dirname "$0")"

REPO=https://github.com/NVIDIA/NeMo-Speech.cpp
# Pinned in NeMo-Speech.cpp models/index.json, which the CLI also verifies against.
MODEL=nemotron-3.5-asr-streaming-0.6b.q8_0.gguf
MODEL_REPO=nvidia/nemotron-3.5-asr-streaming-0.6b
MODEL_REV=1c8deaecc64b91f034d73e08dd8b64625eb3395d
MODEL_SHA=a5c435f294eea8f88ce68dd27b8c3bfea7f777cb2fbba04fcd30eaa555f429ae

# git-lfs is not required: the only LFS files are the Mandarin TTS tokenizer
# tables, and this build has ENABLE_TTS_ZH=OFF. Without these three filter
# overrides `git checkout` aborts on a machine that has no git-lfs installed.
if [ ! -d NeMo-Speech.cpp/.git ]; then
  git clone --depth 1 \
    -c filter.lfs.smudge=cat -c filter.lfs.process= -c filter.lfs.required=false \
    "$REPO" NeMo-Speech.cpp
fi

# ggml/llama.cpp/cpp-httplib are compiled; the rest are copied only so the
# Dockerfile's unconditional LICENSE staging step finds them.
git -C NeMo-Speech.cpp submodule update --init --depth 1 \
  ggml llama.cpp third_party/cpp-httplib \
  proto/riva-common third_party/flashlight-text third_party/kenlm

mkdir -p models
if [ ! -f "models/$MODEL" ]; then
  curl -fL --retry 3 -o "models/$MODEL" \
    "https://huggingface.co/$MODEL_REPO/resolve/$MODEL_REV/$MODEL"
fi
echo "$MODEL_SHA  models/$MODEL" | shasum -a 256 -c -

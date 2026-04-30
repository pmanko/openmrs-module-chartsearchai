#!/bin/sh
# When started as root (deployments without a separate init container that
# chowns the volume), heal pre-Apr-27 root-owned contents and drop to the
# openmrs user. The application itself always runs as uid 1001.
if [ "$(id -u)" = "0" ]; then
  chown -R 1001:1001 /openmrs/data 2>/dev/null || true
  exec runuser -u openmrs -- "$0" "$@"
fi

MODEL_DIR="/openmrs/data/chartsearchai"
mkdir -p "$MODEL_DIR"

# Embedding model (all-MiniLM-L6-v2, ~86MB)
ONNX_FILE="$MODEL_DIR/model.onnx"
VOCAB_FILE="$MODEL_DIR/vocab.txt"
HF_EMBED="https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main"

if [ ! -f "$ONNX_FILE" ]; then
  echo "Downloading all-MiniLM-L6-v2 ONNX model (~86MB)..."
  curl -fsSL -o "$ONNX_FILE" "$HF_EMBED/onnx/model.onnx"
  echo "Embedding model downloaded."
fi

if [ ! -f "$VOCAB_FILE" ]; then
  echo "Downloading all-MiniLM-L6-v2 vocab..."
  curl -fsSL -o "$VOCAB_FILE" "$HF_EMBED/vocab.txt"
  echo "Vocab downloaded."
fi

# LLM model (Gemma 4 E4B Instruct Q4_K_M, ~5GB)
# Save with the filename the module's default config expects.
# Background the download and resume on container restart via curl -C - so
# OpenMRS can become healthy without waiting for the transfer; deploy
# health-poll loops time out at ~5min, but the actual download can take
# longer on slow networks. Chart search queries return errors until the
# .partial file is renamed to its final name.
LLM_FILE="$MODEL_DIR/gemma-4-E4B-it-Q4_K_M.gguf"
LLM_PARTIAL="$LLM_FILE.partial"
HF_LLM="https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-E4B-it-Q4_K_M.gguf"

if [ ! -f "$LLM_FILE" ]; then
  if [ -f "$LLM_PARTIAL" ]; then
    echo "Resuming Gemma 4 E4B Q4_K_M download (~5GB) in background..."
  else
    echo "Starting Gemma 4 E4B Q4_K_M download (~5GB) in background; chart search will be unavailable until it completes..."
  fi
  (
    # --speed-time/--speed-limit aborts if avg throughput stays under 1 KB/s
    # for 60s, so a stalled TCP connection (Hugging Face hangs the socket
    # without closing it) doesn't leave curl waiting on a dead peer
    # indefinitely. The script already retries on the next container start
    # via curl -C -, so a transient abort is self-healing.
    if curl -fsSL -C - --speed-time 60 --speed-limit 1024 -o "$LLM_PARTIAL" "$HF_LLM"; then
      mv "$LLM_PARTIAL" "$LLM_FILE"
      echo "LLM model downloaded."
    else
      echo "LLM model download failed; will retry on next container start."
    fi
  ) &
fi

exec /openmrs/startup.sh

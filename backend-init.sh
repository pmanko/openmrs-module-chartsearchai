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

# LLM model (Gemma 4 26B MoE Instruct UD-Q4_K_M, ~17GB)
# Save with the filename the module's default config expects
LLM_FILE="$MODEL_DIR/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"
HF_LLM="https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF/resolve/main/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"

if [ ! -f "$LLM_FILE" ]; then
  echo "Downloading Gemma 4 26B MoE UD-Q4_K_M (~17GB) — this may take a while..."
  curl -fsSL -o "$LLM_FILE" "$HF_LLM"
  echo "LLM model downloaded."
fi

exec /openmrs/startup.sh

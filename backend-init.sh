#!/bin/sh
MODEL_DIR="/openmrs/data/chartsearchai"
mkdir -p "$MODEL_DIR"

# Embedding model (all-MiniLM-L6-v2, ~86MB)
ONNX_FILE="$MODEL_DIR/model.onnx"
VOCAB_FILE="$MODEL_DIR/vocab.txt"
HF_EMBED="https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main"
DOWNLOADED_FILE=0

if [ ! -f "$ONNX_FILE" ]; then
  echo "Downloading all-MiniLM-L6-v2 ONNX model (~86MB)..."
  curl -fsSL -o "$ONNX_FILE" "$HF_EMBED/onnx/model.onnx"
  echo "Embedding model downloaded."
  DOWNLOADED_FILE=1
fi

if [ ! -f "$VOCAB_FILE" ]; then
  echo "Downloading all-MiniLM-L6-v2 vocab..."
  curl -fsSL -o "$VOCAB_FILE" "$HF_EMBED/vocab.txt"
  echo "Vocab downloaded."
  DOWNLOADED_FILE=1
fi

# LLM model (MedGemma 1.5 4B Q4_K_M, ~2.5GB)
# Save with the filename the module's default config expects
LLM_FILE="$MODEL_DIR/medgemma-1.5-4b-it.Q4_K_M.gguf"
HF_LLM="https://huggingface.co/unsloth/medgemma-1.5-4b-it-GGUF/resolve/main/medgemma-1.5-4b-it-Q4_K_M.gguf"

if [ ! -f "$LLM_FILE" ]; then
  echo "Downloading MedGemma 1.5 4B Q4_K_M (~2.5GB) — this may take a few minutes..."
  curl -fsSL -o "$LLM_FILE" "$HF_LLM"
  echo "LLM model downloaded."
  DOWNLOADED_FILE=1
fi

test $DOWNLOADED_FILE -eq 1 && chown -R 1001:1001 "$MODEL_DIR"

exec /openmrs/startup.sh

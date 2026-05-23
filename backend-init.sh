#!/bin/sh
# Pre-fetch the ML model files this image needs into the OpenMRS appdata
# volume:
#   querystore/model.onnx ......... e5-base-v2 sentence embedder (~440MB)
#   querystore/vocab.txt .......... BERT WordPiece vocab for e5
#   chartsearchai/<gguf-file> ..... local LLM weights — Gemma 4 E4B (~5GB),
#                                   with Gemma 4 E2B (~3GB) also pulled so
#                                   an operator can A/B latency between the
#                                   two by flipping chartsearchai.llm.modelFilePath
#
# Architectural note: chartsearchai delegates retrieval to querystore via
# `chartsearchai.querystore.enabled=true` (the recommended deployment).
# That path uses the embedder under querystore/ for top-K retrieval, with
# the local LLM doing the filtering in the answer phase. chartsearchai's
# own embedding-pre-filter pipeline (querystore=false, preFilter=true) is
# being deprecated — its filter thresholds are L6-v2-distribution-specific
# and don't transfer to other embedders without re-tuning — so we don't
# provision a chartsearchai-side embedder here.
#
# Operator wiring (global properties) required for fresh deployments —
# both modules ship these with empty/false defaults, so the files this
# script provisions are unused until an operator points the GPs at them:
#   chartsearchai.querystore.enabled    = true
#   querystore.embedding.modelFilePath  = querystore/model.onnx
#   querystore.embedding.vocabFilePath  = querystore/vocab.txt
#   querystore.embedding.queryModelFilePath = (empty — single encoder)
# A follow-up change will move those defaults into querystore's config.xml
# so a fresh deploy works without manual GP wiring; until then, the
# project README's deployment section is the source of truth.

# When started as root (deployments without a separate init container that
# chowns the volume), heal pre-Apr-27 root-owned contents and drop to the
# openmrs user. The application itself always runs as uid 1001.
if [ "$(id -u)" = "0" ]; then
  chown -R 1001:1001 /openmrs/data 2>/dev/null || true
  exec runuser -u openmrs -- "$0" "$@"
fi

QS_DIR="/openmrs/data/querystore"
LLM_DIR="/openmrs/data/chartsearchai"
mkdir -p "$QS_DIR" "$LLM_DIR"

# ---- embedder: e5-base-v2 ---------------------------------------------------
# e5-base-v2 was chosen empirically over the alternatives we tested
# (sentence-transformers/all-MiniLM-L6-v2 and ncbi/MedCPT-* article+query
# encoders). On colloquial clinical questions a chart-search user actually
# types — "any heart issues?", "any cardiovascular issues?" — it bridges
# to the formal clinical record terms ("Cerebrovascular Accident") that
# L6-v2's tighter clusters miss and that MedCPT's PubMed-trained clusters
# pedantically refuse to associate. The 50% divergence we measured
# against chartsearchai's L6-v2-tuned eval baseline is the LLM-as-filter
# design absorbing what L6-v2 used to do at the retrieval layer; on the
# querystore + LLM path that's a property, not a quality regression.
#
# Source is Xenova's ONNX export rather than `intfloat/e5-base-v2`'s
# canonical repo: Xenova ships a self-contained model.onnx file, while
# the canonical repo's onnx/ subdirectory uses external-data format
# (graph file + separate `model.onnx_data` weights sidecar). Downloading
# only the graph file produces a tiny "successful" file that the ONNX
# runtime can open but cannot execute — the bug shape that broke the
# previous L6-v2 download. The size guard below is the second line of
# defense against any future upstream format change.
ONNX_FILE="$QS_DIR/model.onnx"
VOCAB_FILE="$QS_DIR/vocab.txt"
HF_EMBED="https://huggingface.co/Xenova/e5-base-v2/resolve/main"
ONNX_MIN_BYTES=200000000  # ~200MB; well under e5's ~440MB self-contained size
VOCAB_MIN_BYTES=200000    # ~200KB; well under e5's ~230KB vocab.txt

if [ ! -f "$ONNX_FILE" ]; then
  echo "Downloading e5-base-v2 ONNX model (~440MB)..."
  # --speed-time/--speed-limit abort a stalled connection after 60s rather
  # than hanging the container start indefinitely; a partial file then
  # trips the size guard below, which rm's it so the next start retries
  # from zero.
  curl -fsSL --speed-time 60 --speed-limit 1024 -o "$ONNX_FILE" "$HF_EMBED/onnx/model.onnx"
fi

# Size guard. A graph-only ONNX file from an external-data export is
# ~1MB and the runtime fails late, at first inference, with a misleading
# "Not a directory" error trying to read a sidecar weights file that
# doesn't exist next to the graph. Catch that at download time.
ONNX_BYTES=$(stat -c %s "$ONNX_FILE" 2>/dev/null || stat -f %z "$ONNX_FILE" 2>/dev/null || echo 0)
if [ "$ONNX_BYTES" -lt "$ONNX_MIN_BYTES" ]; then
  echo "ERROR: $ONNX_FILE is only ${ONNX_BYTES} bytes (expected at least ${ONNX_MIN_BYTES})." >&2
  echo "       The HuggingFace ONNX export likely changed shape upstream." >&2
  echo "       Remove $ONNX_FILE and re-run; if the file is still small," >&2
  echo "       the source repo's onnx/model.onnx is now external-data format" >&2
  echo "       and the sidecar (model.onnx_data) must be staged alongside it." >&2
  rm -f "$ONNX_FILE"
  exit 1
fi
echo "Embedder ready: $ONNX_FILE (${ONNX_BYTES} bytes)."

if [ ! -f "$VOCAB_FILE" ]; then
  echo "Downloading e5-base-v2 vocab..."
  curl -fsSL --speed-time 60 --speed-limit 1024 -o "$VOCAB_FILE" "$HF_EMBED/vocab.txt"
fi

# Vocab size guard, parallel to the ONNX one. e5-base-v2's WordPiece vocab
# is ~230KB. A short or empty vocab would either fail tokenizer init or,
# worse, silently degrade embeddings as missing tokens fall back to [UNK]
# — both modes are easier to diagnose at startup than at first query.
VOCAB_BYTES=$(stat -c %s "$VOCAB_FILE" 2>/dev/null || stat -f %z "$VOCAB_FILE" 2>/dev/null || echo 0)
if [ "$VOCAB_BYTES" -lt "$VOCAB_MIN_BYTES" ]; then
  echo "ERROR: $VOCAB_FILE is only ${VOCAB_BYTES} bytes (expected at least ${VOCAB_MIN_BYTES})." >&2
  echo "       Truncated vocab fails tokenizer init or silently degrades to [UNK] fallbacks." >&2
  echo "       Remove the file and re-run; if it stays small, the upstream export changed." >&2
  rm -f "$VOCAB_FILE"
  exit 1
fi
echo "Vocab ready: $VOCAB_FILE (${VOCAB_BYTES} bytes)."

# ---- LLM: Gemma 4 E4B Instruct, Q4_K_M -------------------------------------
# Chosen over Gemma 4 E2B, Gemma 4 26B-MoE, and Llama-3.2-3B as the
# smallest model that reliably filters querystore's top-K candidates
# without the small-LLM failure modes we observed:
#   - Llama-3.2-3B collapsed pulse readings into "HB" values when given a
#     full chart, and over-trusted retrieval prior in querystore mode
#     (cited "Cocaine abuse" as a medication).
#   - Gemma 4 E2B over-cited (included benign neoplasms in cancer answers,
#     listed 18 BP readings as cardiovascular records).
#   - Gemma 4 26B-MoE is marginally cleaner on edge cases but its ~16GB
#     footprint isn't justified given E4B's strength on this question class.
# E4B holds the benign/malignant distinction, keeps citations focused on
# clinically-relevant records, and fits in ~5GB of memory.
#
# Background the download and resume on container restart via `curl -C -`
# so OpenMRS can become healthy without waiting for the transfer; deploy
# health-poll loops time out at ~5min, but the actual download can take
# longer on slow networks. Chart search queries return errors until the
# .partial file is renamed to its final name.
# Inner worker: actually performs the download and the .partial→final
# rename. Receives all required values as positional args so each
# backgrounded invocation has its own argument snapshot — reading globals
# from a backgrounded subshell would race with the parent shell's next
# fetch_llm_in_background call overwriting them.
_download_llm_file() {
  _url=$1
  _partial=$2
  _target=$3
  _label=$4
  # --speed-time/--speed-limit aborts if avg throughput stays under 1 KB/s
  # for 60s, so a stalled TCP connection (Hugging Face hangs the socket
  # without closing it) doesn't leave curl waiting on a dead peer
  # indefinitely. On the next container start (operator-initiated under
  # the default restart=no policy), curl -C - resumes from .partial.
  if curl -fsSL -C - --speed-time 60 --speed-limit 1024 -o "$_partial" "$_url"; then
    mv "$_partial" "$_target"
    echo "$_label downloaded: $_target"
  else
    echo "$_label download failed; restart the backend container to retry (curl -C - resumes from the .partial file)."
  fi
}

# Helper: emit the start/resume log line and background the actual
# download. $1 url, $2 filename (under $LLM_DIR), $3 human label, $4 size
# hint, $5 availability-note fragment appended to the log line — callers
# pass the served-vs-standby wording so the helper itself doesn't encode
# which model is currently active.
#
# Each invocation backgrounds, so two calls run in parallel — total volume
# need on /openmrs/data is now ~8GB (E4B ~5GB + E2B ~3GB).
fetch_llm_in_background() {
  url=$1
  filename=$2
  label=$3
  size_hint=$4
  availability_note=$5
  target="$LLM_DIR/$filename"
  partial="$target.partial"
  if [ -f "$target" ]; then
    return
  fi
  if [ -f "$partial" ]; then
    echo "Resuming $label download (${size_hint}) in background${availability_note}..."
  else
    echo "Starting $label download (${size_hint}) in background${availability_note}..."
  fi
  _download_llm_file "$url" "$partial" "$target" "$label" &
}

# E4B is the default served model (config.xml defaults
# chartsearchai.llm.modelFilePath to gemma-4-E4B-it-Q4_K_M.gguf). E2B is
# provisioned alongside as a standby so an operator can A/B latency by
# flipping the GP between the two filenames and waiting for llama-server's
# idle-restart to pick up the new weights — no redeploy required.
fetch_llm_in_background \
  "https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-E4B-it-Q4_K_M.gguf" \
  "gemma-4-E4B-it-Q4_K_M.gguf" \
  "Gemma 4 E4B Q4_K_M" \
  "~5GB" \
  "; chart search will be unavailable until it completes"

fetch_llm_in_background \
  "https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-Q4_K_M.gguf" \
  "gemma-4-E2B-it-Q4_K_M.gguf" \
  "Gemma 4 E2B Q4_K_M" \
  "~3GB" \
  " (standby model — chart search keeps using the currently-served weights)"

exec /openmrs/startup.sh

#!/usr/bin/env bash
set -e

repo=${VALEN_REPO:-Valen-Team/Valen-Decision-1.5-0.8B-ONNX}
target=${1:-models/${repo#*/}}
mkdir -p "$target"
curl -fL "https://huggingface.co/$repo/resolve/main/valen_onnx.json" -o "$target/valen_onnx.json"
jq -r '(["config.json", "tokenizer.json", "chat_template.jinja"] + .variants.fp32.components.embedding + .variants.fp32.components.decision) | unique[]' "$target/valen_onnx.json" |
  while IFS= read -r path; do
    mkdir -p "$(dirname "$target/$path")"
    curl -fL "https://huggingface.co/$repo/resolve/main/$path" -o "$target/$path"
  done

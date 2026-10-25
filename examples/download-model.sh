#!/usr/bin/env bash
set -e

repo=Valen-Team/Valen-0.8B-ONNX
target=${1:-models/Valen-0.8B-ONNX}
mkdir -p "$target"
curl -fL "https://huggingface.co/$repo/resolve/main/valen_onnx.json" -o "$target/valen_onnx.json"
jq -r '(["LICENSE", "config.json", "tokenizer.json", "chat_template.jinja"] + .variants.fp32.components.embedding + .variants.fp32.components.decision) | unique[]' "$target/valen_onnx.json" |
  while IFS= read -r path; do
    mkdir -p "$(dirname "$target/$path")"
    curl -fL "https://huggingface.co/$repo/resolve/main/$path" -o "$target/$path"
  done

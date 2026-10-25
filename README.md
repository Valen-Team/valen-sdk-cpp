<p align="center">
  <a href="https://github.com/Valen-Team/Valen"><img src="assets/branding/valen-mark-white.png" alt="Valen" width="112"></a>
</p>
<h1 align="center">Valen C++ text integration</h1>
<p align="center"><strong>Fast Decisions. Continuous Evolution.</strong></p>
<p align="center">Run Valen’s System One text decisions in a native C++ process.</p>

<p align="center">
  <strong>English</strong> · <a href="README_zh.md">简体中文</a><br>
  <a href="https://github.com/Valen-Team/Valen">Valen</a> · <a href="https://valen-team.github.io/">Try Valen</a> · <a href="https://valen-team.github.io/docs/">Documentation</a> · <a href="https://huggingface.co/Valen-Team">Models</a>
</p>

<p align="center">
  <a href="https://github.com/Valen-Team/valen-sdk-python">Python SDK</a> · <a href="https://github.com/Valen-Team/valen-sdk-js">JavaScript SDK</a> · <strong>C++ integration</strong>
</p>

[Valen](https://github.com/Valen-Team/Valen) is a multimodal System One decision model that returns structured decision probabilities for your tasks. This repository is the Valen family’s native C++ text integration example.

Run Valen text decisions inside a C++ process and turn natural language into categories, yes/no probabilities and scores. This repository provides a complete command-line integration example: read SystemOne JSON, compile text and call ONNX Runtime directly. Use the `Model` and text compiler in [main.cpp](main.cpp) as a starting point for a native application.

The current entry point supports **SystemOne text, FP32 CPU and one request per process**. For images, video, Decisions or HTTP, use the [Python SDK](https://github.com/valen-team/valen-sdk-python) or [JavaScript SDK](https://github.com/valen-team/valen-sdk-js).

## Prerequisites

Use a C++17 compiler, CMake 3.19+, [Rust/Cargo](https://www.rust-lang.org/tools/install), Git, curl and jq. Rust builds the tokenizer and template static library; inference runs directly in the C++ process.

Download the code and 0.8B CPU text model files:

```bash
git clone https://github.com/valen-team/valen-sdk-cpp.git
cd valen-sdk-cpp
bash examples/download-model.sh
```

[The download script](examples/download-model.sh) selects only the graphs, weights and configuration required for text inference and saves them under `models/Valen-0.8B-ONNX`. It skips vision and other precisions. For larger models, see the [2B](https://huggingface.co/Valen-Team/Valen-2B-ONNX) and [4B](https://huggingface.co/Valen-Team/Valen-4B-ONNX) FP32 variants.

## Build

### macOS Apple Silicon

```bash
curl -L https://github.com/microsoft/onnxruntime/releases/download/v1.31.0/onnxruntime-osx-arm64-1.31.0.tgz -o onnxruntime.tgz
tar -xzf onnxruntime.tgz
cargo build --release --manifest-path native/Cargo.toml
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DORT_LIBRARY="$PWD/onnxruntime-osx-arm64-1.31.0/lib/libonnxruntime.1.31.0.dylib"
cmake --build build --config Release
```

### Linux x86-64

```bash
curl -L https://github.com/microsoft/onnxruntime/releases/download/v1.31.0/onnxruntime-linux-x64-1.31.0.tgz -o onnxruntime.tgz
tar -xzf onnxruntime.tgz
cargo build --release --manifest-path native/Cargo.toml
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DORT_LIBRARY="$PWD/onnxruntime-linux-x64-1.31.0/lib/libonnxruntime.so.1.31.0"
cmake --build build --config Release
```

These commands use [official ONNX Runtime binaries](https://github.com/microsoft/onnxruntime/releases/tag/v1.31.0). Real-model validation covers 0.8B on macOS ARM; the Linux build has not been tested.

## Run

```bash
./build/valen-cpp models/Valen-0.8B-ONNX examples/request.json
```

[request.json](examples/request.json) asks a choice, yes/no and score question about one customer message. The program prints JSON with these fields:

| Field | Meaning |
| --- | --- |
| `answers.intent.choice` | `address` or `cancel`, for routing the request |
| `answers.change_address.noul` | Probability of an address-change request, from 0 to 1 |
| `answers.explicitness.score` | Request explicitness, from 0 to 2 |

Edit `state`, question instructions and candidates to use your own task. `execution` supports `shared_state` and `question`. Read a request from stdin:

```bash
cat examples/request.json | ./build/valen-cpp models/Valen-0.8B-ONNX -
```

## Source and license

[main.cpp](main.cpp) contains text compilation, ORT inference and result handling. [native/](native) uses Hugging Face tokenizers and MiniJinja for tokenization and template rendering. [vendor/](vendor) contains the ORT headers and JSON parser.

Valen is Apache-2.0; see [LICENSE](LICENSE). ORT and nlohmann/json are MIT; tokenizers and MiniJinja are Apache-2.0, with licenses distributed with their source or dependencies.

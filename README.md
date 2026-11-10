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

Run Valen text decisions inside a C++ process and turn natural language into categories, yes/no probabilities and scores. This repository provides a complete command-line integration example: read SystemOne JSON, compile text and call ONNX Runtime directly. Use the `Model` and text compiler in [valen.hpp](valen.hpp) as a starting point for a native application.

The current entry point supports **SystemOne text, FP32 CPU and one request per process**, and runs both Valen-Decision-1.5 and Valen-Decision-1.0 ONNX bundles: it reads the model generation from the bundle's `config.json`, so the same binary serves either. For images, video, Decisions or HTTP, use the [Python SDK](https://github.com/valen-team/valen-sdk-python) or [JavaScript SDK](https://github.com/valen-team/valen-sdk-js).

## Complete application tutorials

Complete workflows are maintained in the [official Cookbook](https://valen-team.github.io/cookbook/): [ticket routing](https://valen-team.github.io/cookbook/ticket-routing/), [when no option fits](https://valen-team.github.io/cookbook/unknown-categories/), [product-image screening](https://valen-team.github.io/cookbook/product-screening/) and [video tagging](https://valen-team.github.io/cookbook/video-tagging/). Code, samples and measured outputs share one source. This repository retains SDK interface and installation examples. In the Python/JavaScript SDKs, `Valen.local` loads ONNX and `Valen.remote` connects to GGUF HTTP services. C++ support remains scoped by this repository's reference.


## Prerequisites

Use a C++17 compiler, CMake 3.19+, [Rust/Cargo](https://www.rust-lang.org/tools/install), Git, curl and jq. Rust builds the tokenizer and template static library; inference runs directly in the C++ process.

Download the code and the Valen-Decision-1.5 0.8B CPU text model files:

```bash
git clone https://github.com/valen-team/valen-sdk-cpp.git
cd valen-sdk-cpp
bash examples/download-model.sh
```

[The download script](examples/download-model.sh) selects only the graphs, weights and configuration required for text inference and saves them under `models/Valen-Decision-1.5-0.8B-ONNX`. It skips vision, grounding and other precisions. Set `VALEN_REPO` to fetch another bundle, for example a larger Valen-Decision-1.5 size or a Valen-Decision-1.0 model:

```bash
VALEN_REPO=Valen-Team/Valen-Decision-1.5-2B-ONNX bash examples/download-model.sh
```

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

These commands use [official ONNX Runtime binaries](https://github.com/microsoft/onnxruntime/releases/tag/v1.31.0). The build also produces `valen-cpp-tests`, which checks the Valen-Decision-1.5 compiler segments, block attention masks and Noul criteria; run it with `ctest --test-dir build`. Real-model validation covers Valen-Decision-1.5 0.8B on macOS ARM, matching the Python SDK; the Linux build has not been tested.

## Run

```bash
./build/valen-cpp models/Valen-Decision-1.5-0.8B-ONNX examples/request.json
```

[request.json](examples/request.json) asks a choice, yes/no and score question about one customer message. The program prints JSON with these fields:

| Field | Meaning |
| --- | --- |
| `answers.intent.choice` | `address` or `cancel`, for routing the request |
| `answers.change_address.noul` | Probability of an address-change request, from 0 to 1 |
| `answers.explicitness.score` | Request explicitness, from 0 to 2 |

Edit `state`, question instructions and candidates to use your own task. With Valen-Decision-1.5, a `noul` question can also carry `criteria` with `true` and `false` descriptions, which the compiler folds into the instructions. `execution` is `shared_state`, where every question reads one shared sequence under block attention; Valen-Decision-1.0 bundles also accept `question`. Read a request from stdin:

```bash
cat examples/request.json | ./build/valen-cpp models/Valen-Decision-1.5-0.8B-ONNX -
```

## Source and license

[valen.hpp](valen.hpp) contains text compilation, the Valen-Decision-1.5 block attention mask, ORT inference and result handling; [main.cpp](main.cpp) is the command-line entry point and [tests/](tests) holds the compiler tests. [native/](native) uses Hugging Face tokenizers and MiniJinja for tokenization and template rendering. [vendor/](vendor) contains the ORT headers and JSON parser.

Valen is Apache-2.0; see [LICENSE](LICENSE). ORT and nlohmann/json are MIT; tokenizers and MiniJinja are Apache-2.0, with licenses distributed with their source or dependencies.

User guides are maintained only on the website: [Documentation](https://valen-team.github.io/docs/) · [Fine-tuning](https://valen-team.github.io/docs/valen/training/) · [Probabilities](https://valen-team.github.io/docs/valen/probabilities/) · [Cookbook](https://valen-team.github.io/cookbook/).

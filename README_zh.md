<p align="center">
  <a href="https://github.com/Valen-Team/Valen"><img src="assets/branding/valen-mark-white.png" alt="Valen" width="112"></a>
</p>
<h1 align="center">Valen C++ text integration</h1>
<p align="center"><strong>快速决策，持续进化。</strong></p>
<p align="center">在原生 C++ 进程中运行 Valen 的 System One 文本决策。</p>

<p align="center">
  <a href="README.md">English</a> · <strong>简体中文</strong><br>
  <a href="https://github.com/Valen-Team/Valen">Valen 主仓库</a> · <a href="https://valen-team.github.io/zh-Hans/">在线体验</a> · <a href="https://valen-team.github.io/zh-Hans/docs/">文档</a> · <a href="https://huggingface.co/Valen-Team">模型</a>
</p>

<p align="center">
  <a href="https://github.com/Valen-Team/valen-sdk-python">Python SDK</a> · <a href="https://github.com/Valen-Team/valen-sdk-js">JavaScript SDK</a> · <strong>C++ 集成</strong>
</p>

[Valen（万澜）](https://github.com/Valen-Team/Valen) 是多模态 System One 决策模型，面向给定任务输出结构化决策概率。本仓库是 Valen 系列的原生 C++ 文本集成示例。

在 C++ 进程中运行 Valen 的文本决策，将自然语言转换为类别、是非概率和评分。本仓库提供完整的命令行集成示例：读取 SystemOne JSON，编译文本并直接调用 ONNX Runtime。可将 [main.cpp](main.cpp) 中的 `Model` 与文本编译代码作为原生应用集成的起点。

当前入口是 **SystemOne 文本、FP32 CPU、每次进程处理一个请求**。图片、视频、Decisions 和 HTTP 接入使用 [Python SDK](https://github.com/valen-team/valen-sdk-python) 或 [JavaScript SDK](https://github.com/valen-team/valen-sdk-js)。

## 完整应用教程

完整业务流程只在[官网 Cookbook](https://valen-team.github.io/zh-Hans/cookbook/)维护：[工单分流](https://valen-team.github.io/zh-Hans/cookbook/ticket-routing/)、[候选项都不对](https://valen-team.github.io/zh-Hans/cookbook/unknown-categories/)、[商品图片筛选](https://valen-team.github.io/zh-Hans/cookbook/product-screening/)、[短视频标注](https://valen-team.github.io/zh-Hans/cookbook/video-tagging/)。代码、样例和实测结果来自同一维护源。本仓库保留 SDK 接口与安装示例；Python/JavaScript SDK 的 `Valen.local` 是 ONNX 路径，GGUF 服务使用其远程 HTTP 客户端。C++ 集成的支持范围以本仓库说明为准。


## 准备

需要 C++17 编译器、CMake 3.19+、[Rust/Cargo](https://www.rust-lang.org/tools/install)、Git、curl 和 jq。Rust 用于构建 tokenizer 和模板处理静态库，推理直接在 C++ 进程中完成。

下载代码和 0.8B 的 CPU 文本模型文件：

```bash
git clone https://github.com/valen-team/valen-sdk-cpp.git
cd valen-sdk-cpp
bash examples/download-model.sh
```

[下载脚本](examples/download-model.sh) 根据模型清单获取文本推理需要的图、权重和配置，保存在 `models/Valen-0.8B-ONNX`；不下载视觉或其他精度的权重。需要更大模型时，可参考 [2B](https://huggingface.co/Valen-Team/Valen-2B-ONNX) 和 [4B](https://huggingface.co/Valen-Team/Valen-4B-ONNX) 的 FP32 变体。

## 构建

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

以上命令使用 [ONNX Runtime 官方发行包](https://github.com/microsoft/onnxruntime/releases/tag/v1.31.0)。本示例已在 macOS ARM 上完成 0.8B 真实模型验证；Linux 构建尚未实测。

## 运行

```bash
./build/valen-cpp models/Valen-0.8B-ONNX examples/request.json
```

[request.json](examples/request.json) 包含同一段客户消息的分类、是非和评分问题。输出为 JSON，主要字段：

| 字段 | 含义 |
| --- | --- |
| `answers.intent.choice` | `address` 或 `cancel`，用于选择业务处理分支 |
| `answers.change_address.noul` | 客户要求改地址的概率，范围 0–1 |
| `answers.explicitness.score` | 请求明确程度，范围 0–2 |

可编辑请求中的 `state`、问题说明和候选类别。`execution` 支持 `shared_state` 和 `question`；程序也可从 stdin 读取请求：

```bash
cat examples/request.json | ./build/valen-cpp models/Valen-0.8B-ONNX -
```

## 代码与许可

[main.cpp](main.cpp) 包含文本编译、ORT 推理和结果处理；[native/](native) 使用 Hugging Face tokenizers 与 MiniJinja 实现分词和模板渲染；[vendor/](vendor) 保存 ORT 头文件与 JSON 解析库。

Valen 为 Apache-2.0，见 [LICENSE](LICENSE)。ORT 和 nlohmann/json 为 MIT；tokenizers、MiniJinja 为 Apache-2.0，许可随依赖或源码提供。

用户指南唯一维护在官网：[Documentation](https://valen-team.github.io/zh-Hans/docs/) · [Fine-tuning](https://valen-team.github.io/zh-Hans/docs/valen/training/) · [Probabilities](https://valen-team.github.io/zh-Hans/docs/valen/probabilities/) · [Cookbook](https://valen-team.github.io/zh-Hans/cookbook/).

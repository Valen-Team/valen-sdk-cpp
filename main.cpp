// Apache-2.0. Native CPU FP32 text example; model execution stays in ONNX Runtime.
#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <nlohmann/json.hpp>
#include <numeric>
#include <onnxruntime_cxx_api.h>
#include <stdexcept>
#include <string>
#include <vector>
using J = nlohmann::ordered_json;
namespace fs = std::filesystem;
extern "C" {
void *valen_components_new(const char *, const char *);
char *valen_components_call(void *, const char *, const char *);
void valen_components_free(void *);
void valen_string_free(char *);
}
void require(bool ok, const std::string &message) {
  if (!ok)
    throw std::runtime_error(message);
}
std::string read(const fs::path &path) {
  std::ifstream f(path, std::ios::binary);
  require(bool(f), "Cannot open " + path.string());
  return {std::istreambuf_iterator<char>(f), {}};
}
using Span = std::array<int64_t, 2>;
struct Branch {
  Span context, instruction;
  std::vector<Span> candidates;
  std::vector<int64_t> positions, ids;
  int64_t decision;
};
struct Question {
  std::string id, kind, instruction;
  std::vector<std::string> keys, descriptions;
  std::vector<Branch> branches;
};
struct Compiled {
  std::vector<Question> questions;
  std::vector<int64_t> shared;
  int64_t logical = 0, compute = 0;
};
class Components {
  void *handle_;
  std::vector<std::string> special_;

public:
  explicit Components(const fs::path &path) {
    auto tokenizer = read(path / "tokenizer.json"), templ = read(path / "chat_template.jinja");
    handle_ = valen_components_new(tokenizer.c_str(), templ.c_str());
    require(handle_, "Cannot initialize tokenizer/chat template");
    auto tokenizer_data = J::parse(tokenizer);
    for (const auto &token : tokenizer_data["added_tokens"])
      if (token.value("special", false))
        special_.push_back(token["content"]);
  }
  ~Components() { valen_components_free(handle_); }
  Components(const Components &) = delete;
  J call(const char *op, const std::string &text) {
    char *raw = valen_components_call(handle_, op, text.c_str());
    require(raw, "Native text component failed");
    std::string copied(raw);
    valen_string_free(raw);
    auto result = J::parse(copied);
    require(!result.contains("error"), result.value("error", "Native text component failed"));
    return result;
  }
  std::vector<int64_t> encode(const std::string &text) {
    return call("encode", text)["ids"].get<std::vector<int64_t>>();
  }
  void check(const std::string &text) {
    require(text.find(char(0)) == std::string::npos, "NUL is not supported by this text example");
    for (const auto &token : special_)
      require(text.find(token) == std::string::npos, "Reserved tokenizer control token");
  }
  Span append(std::vector<int64_t> &suffix, int64_t base, const std::string &text, Span span = {-1, -1}) {
    auto encoded = call("encode", text);
    auto ids = encoded["ids"].get<std::vector<int64_t>>();
    int64_t offset = base + suffix.size();
    suffix.insert(suffix.end(), ids.begin(), ids.end());
    if (span[0] < 0)
      return {-1, -1};
    int64_t first = -1, last = -1;
    for (size_t i = 0; i < ids.size(); ++i) {
      const auto &pair = encoded["offsets"][i];
      if (pair[0].get<int64_t>() < span[1] && pair[1].get<int64_t>() > span[0]) {
        if (first < 0)
          first = i;
        last = i;
      }
    }
    require(first >= 0, "Empty role span");
    return {offset + first, offset + last + 1};
  }
  Compiled compile(const J &request, const J &config) {
    require(request.contains("state") && request.contains("questions"),
            "Expected SystemOne state and questions");
    J messages;
    if (request["state"].is_string())
      messages = J::array({{{"role", "user"}, {"content", request["state"]}}});
    else {
      require(request["state"].is_object() && request["state"].contains("messages"),
              "state must be text or messages");
      messages = request["state"]["messages"];
    }
    for (auto &message : messages) {
      std::string role = message.value("role", "");
      require(role == "user" || role == "assistant" || role == "system", "Invalid message role");
      auto &content = message["content"];
      if (content.is_string())
        check(content);
      else {
        require(content.is_array(), "Expected text content");
        for (const auto &part : content) {
          require(part.value("type", "") == "text", "C++ example supports text only");
          check(part.at("text"));
        }
      }
    }
    auto rendered =
        call("render", J({{"messages", messages}, {"add_generation_prompt", false}}).dump())["text"]
            .get<std::string>();
    auto base = encode(rendered);
    int64_t length = base.size();
    require(length > 0, "Empty context");
    require(request["questions"].is_object() && !request["questions"].empty() &&
                request["questions"].size() <= 256,
            "Expected 1-256 questions");
    Compiled out;
    out.logical = length;
    for (auto it = request["questions"].begin(); it != request["questions"].end(); ++it) {
      const auto &q = it.value();
      Question question;
      question.id = it.key();
      question.kind = q.value("type", "");
      question.instruction = q.at("instructions").get<std::string>();
      require(!question.instruction.empty(), "Empty instructions");
      check(question.instruction);
      if (question.kind == "noul") {
        question.keys = {"true", "false"};
        question.descriptions = {"True / 是：满足问题中的条件。", "False / 否：不满足问题中的条件。"};
      } else if (question.kind == "choice") {
        require(q.at("criteria").is_object(), "Choice criteria must be an object");
        for (auto p = q["criteria"].begin(); p != q["criteria"].end(); ++p) {
          question.keys.push_back(p.key());
          question.descriptions.push_back(p.value().get<std::string>());
        }
      } else {
        require(question.kind == "score" && q.at("criteria").is_array() && q["criteria"].size() >= 2,
                "Invalid score question");
        for (size_t i = 0; i < q["criteria"].size(); ++i) {
          question.keys.push_back(std::to_string(i));
          question.descriptions.push_back(q["criteria"][i]);
        }
      }
      require(!question.keys.empty() && question.keys.size() <= 255, "Invalid candidate count");
      for (size_t i = 0; i < question.keys.size(); ++i) {
        require(!question.keys[i].empty() && !question.descriptions[i].empty(), "Empty candidate");
        check(question.keys[i]);
        check(question.descriptions[i]);
      }
      out.questions.push_back(std::move(question));
    }
    auto execution = request.value("execution", config.value("qwen_execution", std::string("shared_state")));
    require(execution == "shared_state" || execution == "question", "Unknown execution mode");
    int64_t limit = config.value("valen_max_length", 32768);
    auto candidate = [&](Question &q, Branch &b, std::vector<int64_t> &suffix, size_t i) {
      std::string prefix = q.kind == "score" ? "" : q.keys[i] + ": ";
      auto text = prefix + q.descriptions[i];
      b.candidates.push_back(append(suffix, length, text, {int64_t(prefix.size()), int64_t(text.size())}));
      b.positions.push_back(length + suffix.size() - 1);
      append(suffix, length, "\n");
    };
    if (execution == "shared_state") {
      std::vector<int64_t> suffix;
      append(suffix, length, "<|im_start|>user\n");
      for (size_t i = 0; i < out.questions.size(); ++i) {
        auto &q = out.questions[i];
        Branch b;
        b.context = {0, length};
        std::string prefix = "Question " + std::to_string(i + 1) + "\nTask: " + q.kind + "\nQuestion: ";
        b.instruction = append(suffix, length, prefix + q.instruction + "\nCandidates:\n",
                               {int64_t(prefix.size()), int64_t(prefix.size() + q.instruction.size())});
        for (size_t k = 0; k < q.keys.size(); ++k)
          candidate(q, b, suffix, k);
        q.branches.push_back(std::move(b));
      }
      append(suffix, length, "<|im_end|>\n<|im_start|>assistant\n");
      for (size_t i = 0; i < out.questions.size(); ++i) {
        append(suffix, length, "Question " + std::to_string(i + 1) + " Decision:");
        out.questions[i].branches[0].decision = length + suffix.size() - 1;
        append(suffix, length, "\n");
      }
      out.shared = base;
      out.shared.insert(out.shared.end(), suffix.begin(), suffix.end());
      require(int64_t(out.shared.size()) <= limit, "Input exceeds max_length; no truncation");
      out.logical = out.compute = out.shared.size();
      for (auto &q : out.questions)
        q.branches[0].ids = out.shared;
    } else {
      for (auto &q : out.questions) {
        size_t groups = q.kind == "score" ? q.keys.size() : 1;
        for (size_t group = 0; group < groups; ++group) {
          std::vector<int64_t> suffix;
          Branch b;
          b.context = {0, length};
          append(suffix, length, "<|im_start|>user\n");
          std::string prefix = "Task: " + q.kind + "\nQuestion: ";
          b.instruction = append(suffix, length, prefix + q.instruction + "\nCandidates:\n",
                                 {int64_t(prefix.size()), int64_t(prefix.size() + q.instruction.size())});
          if (q.kind == "score")
            candidate(q, b, suffix, group);
          else
            for (size_t i = 0; i < q.keys.size(); ++i)
              candidate(q, b, suffix, i);
          append(suffix, length, "Decision:");
          b.ids = base;
          b.ids.insert(b.ids.end(), suffix.begin(), suffix.end());
          b.decision = b.ids.size() - 1;
          require(int64_t(b.ids.size()) <= limit, "Input exceeds max_length; no truncation");
          out.logical += suffix.size();
          out.compute += b.ids.size();
          q.branches.push_back(std::move(b));
        }
      }
    }
    return out;
  }
};
J trace(const Compiled &compiled) {
  J questions = J::array();
  for (const auto &q : compiled.questions) {
    J branches = J::array();
    for (const auto &b : q.branches)
      branches.push_back({{"input_ids", J::array({b.ids})},
                          {"context_span", b.context},
                          {"instruction_span", b.instruction},
                          {"candidate_spans", b.candidates},
                          {"candidate_positions", b.positions},
                          {"decision_position", b.decision}});
    questions.push_back({{"qid", q.id}, {"keys", q.keys}, {"branches", branches}});
  }
  return {
      {"logical_tokens", compiled.logical}, {"compute_tokens", compiled.compute}, {"questions", questions}};
}
J answer(const Question &q, const std::vector<float> &logits, double temperature) {
  require(logits.size() == q.keys.size(), "Wrong candidate logit count");
  double maximum = *std::max_element(logits.begin(), logits.end());
  std::vector<double> probabilities;
  for (auto x : logits) {
    require(std::isfinite(x), "Nonfinite logit");
    probabilities.push_back(std::exp((double(x) - maximum) / temperature));
  }
  double sum = std::accumulate(probabilities.begin(), probabilities.end(), 0.0);
  for (auto &p : probabilities)
    p /= sum;
  auto mode = std::max_element(probabilities.begin(), probabilities.end()) - probabilities.begin();
  J result = {{"type", q.kind}};
  if (q.kind == "noul") {
    result["noul"] = probabilities[0];
    return result;
  }
  result["probabilities"] = J::object();
  for (size_t i = 0; i < q.keys.size(); ++i)
    result["probabilities"][q.keys[i]] = probabilities[i];
  if (q.kind == "choice") {
    result["choice"] = q.keys[mode];
    double chance = 1.0 / probabilities.size();
    result["confidence"] =
        probabilities.size() == 1 ? 1 : std::max(0.0, (probabilities[mode] - chance) / (1 - chance));
  } else {
    double score = 0, distance = 0, uniform = 0;
    result["legend"] = J::object();
    for (size_t i = 0; i < q.keys.size(); ++i) {
      score += i * probabilities[i];
      distance += probabilities[i] * std::abs(double(i) - mode);
      uniform += std::abs(double(i) - (q.keys.size() - 1) / 2.0) / q.keys.size();
      result["legend"][q.keys[i]] = q.descriptions[i];
    }
    result["score"] = score;
    result["confidence"] = std::max(0.0, 1 - distance / uniform);
  }
  return result;
}
class Model {
  fs::path path_;
  J config_, manifest_, variant_;
  Components components_;
  Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "valen-cpp"};
  Ort::SessionOptions options_;
  Ort::AllocatorWithDefaultOptions allocator_;
  std::unique_ptr<Ort::Session> embedding_, decision_;

public:
  explicit Model(fs::path path)
      : path_(std::move(path)), config_(J::parse(read(path_ / "config.json"))),
        manifest_(J::parse(read(path_ / "valen_onnx.json"))), variant_(manifest_["variants"].at("fp32")),
        components_(path_) {
    require(manifest_.value("format_version", 0) == 2, "Expected v2 model bundle");
    options_.SetIntraOpNumThreads(4);
    options_.SetInterOpNumThreads(1);
    options_.DisableCpuMemArena();
    options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  }
  Compiled compile(const J &request) { return components_.compile(request, config_); }
  std::vector<float> run(const std::vector<int64_t> &ids, const std::vector<const Branch *> &branches) {
    if (!embedding_)
      embedding_ = std::make_unique<Ort::Session>(
          env_, (path_ / variant_["graphs"]["embedding"].get<std::string>()).c_str(), options_);
    if (!decision_)
      decision_ = std::make_unique<Ort::Session>(
          env_, (path_ / variant_["graphs"]["decision"].get<std::string>()).c_str(), options_);
    auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<int64_t> ids_shape = {1, int64_t(ids.size())};
    auto id_tensor = Ort::Value::CreateTensor<int64_t>(memory, const_cast<int64_t *>(ids.data()), ids.size(),
                                                       ids_shape.data(), 2);
    const char *e_in[] = {"input_ids"};
    const char *e_out[] = {"inputs_embeds"};
    auto embedded = embedding_->Run(Ort::RunOptions{nullptr}, e_in, &id_tensor, 1, e_out, 1);
    std::vector<int64_t> mask(ids.size(), 1), positions(ids.size() * 3);
    for (size_t i = 0; i < positions.size(); ++i)
      positions[i] = i % ids.size();
    std::vector<int64_t> spans;
    for (auto b : branches)
      for (auto candidate : b->candidates)
        for (auto span : {b->context, b->instruction, candidate, Span{b->decision, b->decision + 1}}) {
          spans.push_back(span[0]);
          spans.push_back(span[1]);
        }
    std::vector<int64_t> pos_shape = {3, 1, int64_t(ids.size())},
                         span_shape = {int64_t(spans.size() / 8), 4, 2};
    std::vector<std::string> names;
    std::vector<Ort::Value> values;
    names.reserve(decision_->GetInputCount());
    values.reserve(decision_->GetInputCount());
    for (size_t i = 0; i < decision_->GetInputCount(); ++i) {
      auto name = decision_->GetInputNameAllocated(i, allocator_);
      names.emplace_back(name.get());
      auto &n = names.back();
      if (n == "inputs_embeds")
        values.push_back(std::move(embedded[0]));
      else if (n == "attention_mask")
        values.push_back(
            Ort::Value::CreateTensor<int64_t>(memory, mask.data(), mask.size(), ids_shape.data(), 2));
      else if (n == "position_ids")
        values.push_back(Ort::Value::CreateTensor<int64_t>(memory, positions.data(), positions.size(),
                                                           pos_shape.data(), 3));
      else if (n == "role_spans")
        values.push_back(
            Ort::Value::CreateTensor<int64_t>(memory, spans.data(), spans.size(), span_shape.data(), 3));
      else {
        require(n.rfind("past", 0) == 0, "Unknown graph input " + n);
        auto type_info = decision_->GetInputTypeInfo(i);
        auto info = type_info.GetTensorTypeAndShapeInfo();
        auto shape = info.GetShape();
        std::vector<const char *> symbolic(shape.size());
        info.GetSymbolicDimensions(symbolic.data(), symbolic.size());
        for (size_t d = 0; d < shape.size(); ++d)
          if (shape[d] < 0) {
            std::string symbol = symbolic[d] ? symbolic[d] : "";
            if (symbol == "batch_size")
              shape[d] = 1;
            else if (symbol == "past_sequence_length")
              shape[d] = 0;
            else if (symbol == "kv_cache_dim")
              shape[d] = config_["text_config"]["head_dim"];
            else
              throw std::runtime_error("Unknown cache dimension " + symbol);
          }
        require(info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                "C++ example requires FP32 graphs");
        auto tensor = Ort::Value::CreateTensor<float>(allocator_, shape.data(), shape.size());
        auto count = tensor.GetTensorTypeAndShapeInfo().GetElementCount();
        std::fill_n(tensor.GetTensorMutableData<float>(), count, 0.0f);
        values.push_back(std::move(tensor));
      }
    }
    std::vector<const char *> pointers;
    for (auto &n : names)
      pointers.push_back(n.c_str());
    const char *output[] = {"logits"};
    auto result =
        decision_->Run(Ort::RunOptions{nullptr}, pointers.data(), values.data(), values.size(), output, 1);
    const float *logits = result[0].GetTensorData<float>();
    auto count = result[0].GetTensorTypeAndShapeInfo().GetElementCount();
    require(count == spans.size() / 8, "Graph candidate count mismatch");
    return {logits, logits + count};
  }
  J predict(const J &request) {
    double temperature = request.value("temperature", 1.0);
    require(!request.contains("temperature") || request["temperature"].is_number(),
            "temperature must be a number");
    require(std::isfinite(temperature) && temperature > 0, "temperature must be finite and positive");
    require(!request.contains("model") || request["model"] == manifest_["model_name"], "Unknown model");
    auto compiled = compile(request);
    J answers = J::object();
    std::vector<float> all;
    size_t offset = 0;
    if (!compiled.shared.empty()) {
      std::vector<const Branch *> branches;
      for (const auto &q : compiled.questions)
        for (const auto &b : q.branches)
          branches.push_back(&b);
      all = run(compiled.shared, branches);
    }
    for (const auto &q : compiled.questions) {
      std::vector<float> logits;
      for (const auto &b : q.branches) {
        if (compiled.shared.empty()) {
          auto result = run(b.ids, {&b});
          logits.insert(logits.end(), result.begin(), result.end());
        } else {
          logits.insert(logits.end(), all.begin() + offset, all.begin() + offset + b.candidates.size());
          offset += b.candidates.size();
        }
      }
      answers[q.id] = answer(q, logits, temperature);
    }
    return {{"model", manifest_["model_name"]},
            {"answers", answers},
            {"usage", {{"input_tokens", compiled.logical}, {"output_tokens", 0}}},
            {"internal_usage", {{"compute_tokens", compiled.compute}}}};
  }
};
int main(int argc, char **argv) {
  try {
    require(argc == 3 || argc == 4, "Usage: valen-cpp MODEL_DIR REQUEST.json [--compile-only]");
    auto body =
        J::parse(std::string(argv[2]) == "-" ? std::string(std::istreambuf_iterator<char>(std::cin), {})
                                             : read(argv[2]));
    if (body.contains("request"))
      body = body["request"];
    Model model(argv[1]);
    bool compile_only = argc == 4 && std::string(argv[3]) == "--compile-only";
    require(argc == 3 || compile_only, "Unknown argument");
    std::cout << (compile_only ? trace(model.compile(body)) : model.predict(body)).dump() << std::endl;
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
}

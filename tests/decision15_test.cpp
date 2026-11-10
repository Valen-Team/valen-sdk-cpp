// Apache-2.0. Valen-Decision-1.5 compiler and block mask tests; mirrors valen-sdk-js test/decision15.test.js.
#include "../valen.hpp"
#include <functional>

static int failures = 0;
#define CHECK(cond)                                                                                          \
  do {                                                                                                       \
    if (!(cond)) {                                                                                           \
      std::cerr << __FILE__ << ":" << __LINE__ << ": CHECK(" #cond ") failed" << std::endl;                  \
      ++failures;                                                                                            \
    }                                                                                                        \
  } while (0)
static void check_throws(const std::function<void()> &f, const std::string &needle, int line) {
  try {
    f();
  } catch (const std::exception &e) {
    if (std::string(e.what()).find(needle) != std::string::npos)
      return;
    std::cerr << "line " << line << ": wrong error: " << e.what() << std::endl;
    ++failures;
    return;
  }
  std::cerr << "line " << line << ": expected error " << needle << std::endl;
  ++failures;
}
#define CHECK_THROWS(expr, needle) check_throws([&] { expr; }, needle, __LINE__)

// One token per UTF-8 byte, so every expected position below can be counted by hand.
static std::string byte_tokenizer() {
  std::vector<int> bytes;
  for (int i = 33; i < 127; ++i)
    bytes.push_back(i);
  for (int i = 161; i < 173; ++i)
    bytes.push_back(i);
  for (int i = 174; i < 256; ++i)
    bytes.push_back(i);
  std::vector<int> codes = bytes;
  for (int value = 0, extra = 0; value < 256; ++value)
    if (std::find(bytes.begin(), bytes.end(), value) == bytes.end()) {
      bytes.push_back(value);
      codes.push_back(256 + extra++);
    }
  auto utf8 = [](int code) {
    std::string out;
    if (code < 0x80)
      out += char(code);
    else {
      out += char(0xC0 | (code >> 6));
      out += char(0x80 | (code & 0x3F));
    }
    return out;
  };
  J vocab = J::object();
  for (size_t i = 0; i < bytes.size(); ++i)
    vocab[utf8(codes[i])] = bytes[i];
  J added = J::array();
  std::vector<std::string> specials = {"<|im_start|>", "<|im_end|>", "<|image_pad|>"};
  for (size_t i = 0; i < specials.size(); ++i)
    added.push_back({{"id", 256 + i},
                     {"content", specials[i]},
                     {"single_word", false},
                     {"lstrip", false},
                     {"rstrip", false},
                     {"normalized", false},
                     {"special", true}});
  J byte_level = {{"type", "ByteLevel"}, {"add_prefix_space", false}, {"trim_offsets", false}, {"use_regex", false}};
  return J({{"version", "1.0"},
            {"truncation", nullptr},
            {"padding", nullptr},
            {"added_tokens", added},
            {"normalizer", {{"type", "NFC"}}},
            {"pre_tokenizer", byte_level},
            {"post_processor", nullptr},
            {"decoder", byte_level},
            {"model",
             {{"type", "BPE"},
              {"dropout", nullptr},
              {"unk_token", nullptr},
              {"continuing_subword_prefix", nullptr},
              {"end_of_word_suffix", nullptr},
              {"fuse_unk", false},
              {"byte_fallback", false},
              {"ignore_merges", false},
              {"vocab", vocab},
              {"merges", J::array()}}}})
      .dump();
}
static const char *kTemplate =
    "{% for message in messages %}<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n{% endfor %}";

using Runs = std::vector<std::array<int64_t, 2>>;
static Runs runs(const std::vector<int64_t> &segments) {
  Runs out;
  for (auto value : segments)
    if (!out.empty() && out.back()[0] == value)
      ++out.back()[1];
    else
      out.push_back({value, 1});
  return out;
}
static std::vector<std::string> visible(const std::vector<float> &mask, size_t length) {
  std::vector<std::string> rows;
  for (size_t q = 0; q < length; ++q) {
    std::string row;
    for (size_t k = 0; k < length; ++k) {
      float value = mask[q * length + k];
      CHECK(value == 0.0f || value == std::numeric_limits<float>::lowest());
      row += value == 0.0f ? '1' : '0';
    }
    rows.push_back(row);
  }
  return rows;
}
static J config(const J &version) {
  J out = J::object();
  if (!version.is_null())
    out["valen_decision_version"] = version;
  return out;
}
static const J kRequest = J::parse(R"({"state":"ab","questions":{
  "flag":{"type":"noul","instructions":"x?"},
  "pick":{"type":"choice","instructions":"yé","criteria":{"a":"A","b":"B"}},
  "rate":{"type":"score","instructions":"z","criteria":["lo","hi"]}}})");

static void shared_state_segments(Components &components) {
  J request = kRequest;
  request["execution"] = "shared_state";
  auto compiled = components.compile(request, config("1.5"));
  CHECK(compiled.prefix == 10);
  CHECK(compiled.questions[0].branches[0].context[1] == 10);
  CHECK(compiled.segments.size() == compiled.shared.size());
  // Generated by valen_sdk.compiler.Compiler._compile_shared with the same tokenizer.
  CHECK(runs(compiled.segments) ==
        (Runs{{-1, 10}, {0, 6}, {1, 146}, {2, 60}, {3, 53}, {0, 13}, {1, 21}, {2, 21}, {3, 21}}));
  std::vector<int64_t> decisions;
  for (const auto &q : compiled.questions)
    decisions.push_back(q.branches[0].decision);
  CHECK(decisions == (std::vector<int64_t>{307, 328, 349}));
  request["execution"] = "question";
  CHECK(components.compile(request, config("1.5")).segments.empty());
}

static void block_mask_modes() {
  std::vector<int64_t> segments = {-1, -1, 0, 1, 1, 2, 0, 1, 2};
  auto rows = [&](const std::string &mode) { return visible(block_attention_mask(mode, segments, 2), 9); };
  using Rows = std::vector<std::string>;
  CHECK(rows("block_bidirectional") == (Rows{"110000000", "110000000", "111111111", "111111111", "111111111",
                                             "111111111", "111111111", "111111111", "111111111"}));
  CHECK(rows("prefix_lm") == (Rows{"110000000", "110000000", "111000000", "111100000", "111110000", "111111000",
                                   "111111100", "111111110", "111111111"}));
  CHECK(rows("question_blocks") == (Rows{"110000000", "110000000", "111000100", "111110110", "111110110",
                                         "111001101", "111000100", "111110110", "111001101"}));
  CHECK(rows("bbox_blocks") == (Rows{"110000000", "110000000", "111000100", "111111111", "111111111", "111111111",
                                     "111000100", "111111111", "111111111"}));
  CHECK_THROWS(block_attention_mask("causal", segments, 2), "Unknown attention mask mode");
}

static void noul_criteria_folding(Components &components) {
  J question = J::parse(
      R"({"type":"noul","instructions":"Is it red?","criteria":{"true":"Mostly red","false":"Any other colour"}})");
  const std::string folded = "Is it red?\nTrue criterion: Mostly red\nFalse criterion: Any other colour";
  CHECK(fold_noul_criteria(question) == J({{"type", "noul"}, {"instructions", folded}}));
  CHECK(question["criteria"] == J({{"true", "Mostly red"}, {"false", "Any other colour"}}));
  J choice = J::parse(R"({"type":"choice","instructions":"Which?","criteria":{"true":"Yes","false":"No"}})");
  J plain = J::parse(R"({"type":"noul","instructions":"Is it red?"})");
  CHECK(fold_noul_criteria(choice) == choice);
  CHECK(fold_noul_criteria(plain) == plain);
  J partial = question;
  partial["criteria"].erase("false");
  CHECK_THROWS(fold_noul_criteria(partial), "Noul criteria must define true and false");
  std::vector<size_t> lengths;
  for (const J &version : {J("1.5"), J("1.0"), J()}) {
    J request = {{"state", "ab"}, {"questions", {{"red", question}}}, {"execution", "shared_state"}};
    auto compiled = components.compile(request, config(version));
    CHECK(compiled.questions[0].instruction == (version == "1.5" ? folded : std::string("Is it red?")));
    CHECK(compiled.questions[0].keys == (std::vector<std::string>{"true", "false"}));
    lengths.push_back(compiled.shared.size());
  }
  CHECK(lengths[0] - lengths[1] == std::string("\nTrue criterion: Mostly red\nFalse criterion: Any other colour").size());
  CHECK(lengths[1] == lengths[2]);
}

static void bbox_blocks_keep_bbox_questions_apart() {
  // state, state | glue | Q1 | Q2 | Q3 | glue | readouts 1, 2, 3; rows as valen_sdk.onnx.block_attention_mask gives them.
  std::vector<int64_t> segments = {-1, -1, 0, 1, 2, 3, 0, 1, 2, 3};
  auto rows = [&](const std::string &mode, std::vector<int64_t> bbox) {
    return visible(block_attention_mask(mode, segments, 2, bbox), 10);
  };
  using Rows = std::vector<std::string>;
  std::string state = "1100000000", glue = "1110001000", merged = "1111111111";
  CHECK(rows("bbox_blocks", {2}) == (Rows{state, state, glue, "1111011101", "1110101010", "1111011101", glue,
                                          "1111011101", "1110101010", "1111011101"}));
  CHECK(rows("bbox_blocks", {}) == (Rows{state, state, glue, merged, merged, merged, glue, merged, merged, merged}));
  std::string one = "1111001100", two = "1110101010", three = "1110011001";
  CHECK(rows("bbox_blocks", {2, 3}) == (Rows{state, state, glue, one, two, three, glue, one, two, three}));
  CHECK(rows("question_blocks", {}) == (Rows{state, state, glue, one, two, three, glue, one, two, three}));
  CHECK(rows("bbox_blocks", {1}) == (Rows{state, state, glue, merged, merged, merged, glue, merged, merged, merged}));
  std::string first = "1111101110";
  CHECK(rows("bbox_blocks", {1, 3}) == (Rows{state, state, glue, first, first, three, glue, first, first, three}));
}

int main() {
  try {
    Components components(byte_tokenizer(), kTemplate);
    shared_state_segments(components);
    block_mask_modes();
    noul_criteria_folding(components);
    bbox_blocks_keep_bbox_questions_apart();
  } catch (const std::exception &e) {
    std::cerr << "unexpected error: " << e.what() << std::endl;
    return 1;
  }
  if (failures) {
    std::cerr << failures << " check(s) failed" << std::endl;
    return 1;
  }
  std::cout << "decision15 tests passed" << std::endl;
  return 0;
}

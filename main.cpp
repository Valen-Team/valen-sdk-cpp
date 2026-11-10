// Apache-2.0. Command-line entry point for the Valen C++ runtime.
#include "valen.hpp"
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

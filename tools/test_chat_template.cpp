// test_chat_template.cpp MODEL_DIR CASES.json -- the server's request parser + renderer on
// each case (an OpenAI chat request body); prints the prompts for a byte diff against the
// checkpoint's own Jinja template (tools/bench/render_jinja.py).
#include "b70/http_request.hpp"
#include <cstdio>
#include <fstream>
#include <sstream>
int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: test_chat_template MODEL_DIR CASES.json\n"); return 2; }
    b70::Tokenizer tk; std::string err;
    if (!tk.load(argv[1], err)) { std::fprintf(stderr, "tokenizer: %s\n", err.c_str()); return 1; }
    std::ifstream f(argv[2]); std::stringstream ss; ss << f.rdbuf();
    const b70::Json cases = b70::Json::parse(ss.str());
    if (argc > 3 && std::string(argv[3]) == "--parse") {   // tool-call output parser
        for (const auto& c : cases.array) {
            const auto calls = b70::parse_xml_tool_calls(c.at("text").text(), c.at("tools").array);
            std::printf("%zu call(s)\n", calls.size());
            for (const auto& k : calls) std::printf("  %s %s\n", k.name.c_str(), k.arguments.c_str());
        }
        return 0;
    }
    std::fprintf(stderr, "supports_tools=%d\n", int(tk.supports_tools()));
    for (const auto& c : cases.array) {
        std::string out;
        try {
            const auto req = b70::parse_completion_request(c.dump(), true);
            out = tk.apply_chat_template(req.messages, req.chat);
        } catch (const std::exception& e) { out = std::string("ERROR: ") + e.what(); }
        std::fwrite(out.data(), 1, out.size(), stdout);
        std::fputs("\n=====CASE=====\n", stdout);
    }
    return 0;
}

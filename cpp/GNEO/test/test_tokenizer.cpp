#include "tokenizer.hpp"
#include <cassert>

int main(int argc, char** argv) {
    // Initialize the tokenizer
    Tokenizer tokenizer("../python/tokenizer/vocab.json");
    tokenizer.parse_vocab();

    auto token = tokenizer.get_token(38333);
    std::cout << "Token for ID 38333: " << token << std::endl;
    assert(token == "opped");
    auto id = tokenizer.get_id("opped");
    assert(id == 38333);
    return 0;
}
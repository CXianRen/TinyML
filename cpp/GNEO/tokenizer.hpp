#pragma once 

#include <map>
#include <utility>
#include <vector>
#include <string>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <iostream>

class Tokenizer {
  private:
    std::map<std::string, int> token_to_id_;
    std::vector<std::string> id_to_token_;
    std::string vocab_file_;
  public:
    Tokenizer(const std::string& vocab_file) : 
      vocab_file_(vocab_file) {}

    std::pair<std::string, int> 
    parser_line(const std::string& line) {
      // remove the "," at the end of the line
      std::string cleaned_line = line;
      if (cleaned_line.back() == ',') {
        cleaned_line.pop_back();
      }
      // split the line by ":"
      size_t colon_pos = cleaned_line.find_last_of(':');
      if (colon_pos == std::string::npos) {
        throw std::runtime_error("Invalid line format: " + line);
      }
      std::string token = cleaned_line.substr(0, colon_pos);
      std::string id_str = cleaned_line.substr(colon_pos + 1);
      // convert \" \\ to 
      for (size_t i = 0; i < token.size(); ++i) {
          if (token[i] == '\\' && i + 1 < token.size()) {
              if (token[i + 1] == '\"') {
                  token[i] = '\"';
                  token.erase(i + 1, 1);
              } else if (token[i + 1] == '\\') {
                  token[i] = '\\';
                  token.erase(i + 1, 1);
              }
          }
      }
      // remove the quotes from the token
      if (token.front() == '\"' && token.back() == '\"') {
        token = token.substr(1, token.size() - 2);
      }
      // convert id_str to int
      // std::cout<< "Parsing token: " << token 
      //           << ", id: " << id_str << std::endl;
      int id = std::stoi(id_str);
      return {token, id};
    }

    void parse_vocab() {
      // open the vocab file and read the tokens
      std::ifstream file(vocab_file_);
      if (!file.is_open()) {
        throw std::runtime_error(
          "Could not open vocab file: " + vocab_file_);  
      }

      std::string line;
      while (std::getline(file, line)) {
        // remove spaces
        line.erase(std::remove(line.begin(), line.end(), ' '), line.end());
        // skip the first line { and the last line }
        if (line.empty() || line == "{" || line == "}") {
          continue;
        }
        // parse the line
        auto token_id_pair = parser_line(line);
        const std::string& token = token_id_pair.first;
        int id = token_id_pair.second;
        token_to_id_[token] = id;
        id_to_token_.push_back(token);
      }
      std::cout << "Parsed " << token_to_id_.size() 
                << " tokens from vocab file: " 
                << vocab_file_ << std::endl;
      file.close();
    }

    int get_id(const std::string& token) const {
      auto it = token_to_id_.find(token);
      if (it != token_to_id_.end()) {
        return it->second;
      } else {
        throw std::runtime_error("Token not found: " + token);
      }
    }

    std::string get_token(int id) const {
      if (id < 0 || id >= static_cast<int>(id_to_token_.size())) {
        throw std::runtime_error("ID out of range: " + std::to_string(id));
      }
      std::string token_str = id_to_token_[id];
      // replace special character Ġ with space
      std::string target = "Ġ";
      size_t pos;
      while ((pos = token_str.find(target)) != std::string::npos) {
          token_str.replace(pos, target.length(), " ");
      }
      return token_str;
    }
};
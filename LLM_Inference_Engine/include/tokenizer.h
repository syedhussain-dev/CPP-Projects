#include <string>
#include <optional>
#include <filesystem>
#include <iostream>
#include <vector>
#include <regex>
#include <fstream>
#include <ios>
#include <algorithm>
#include <sentencepiece_processor.h>


class Tokenizer{
public:
    Tokenizer(std::optional<std::string> tokenizer_model=std::nullopt);
    std::vector<int> encode(std::string s, bool bos, bool eos);
    std::string decode(std::vector<int> t);
    void exporter();

private:
    std::string model_path;
    int n_words;
    int bos_id;
    int eos_id;
    int pad_id;
    sentencepiece::SentencePieceProcessor processor;
    void checkPath(std::string& model_path);

};
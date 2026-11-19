#ifndef TOKENIZER_H
#define TOKENIZER_H
#include <string>
#include <filesystem>
#include <optional>
#include <vector>
#include <sentencepiece_processor.h>

class Tokenizer{
public:
    Tokenizer(std::optional<std::string> tokenizer_model=std::nullopt);
    std::vector<int> encode(std::string s, bool bos, bool eos);
    std::string decode(const std::vector<int>& t);
    void exporter();

private:
    std::string model_path;
    int n_words;
    int bos_id;
    int eos_id;
    int pad_id;
    sentencepiece::SentencePieceProcessor processor;
    void checkPath(std::filesystem::path& checkPath);

};

#endif
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
std::string TOKENIZER_MODEL = "tokenizer.model";


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

Tokenizer::Tokenizer(std::optional<std::string> tokenizer_model=std::nullopt){
    /*
    Constructor of the class. If a tokenizer model is specified then it's used
    else the tokenizer model file is used. The path is checked for validation
    a sentence processor is loaded using the tokenizer file and all private
    variables are defined.

    @param tokenizer_model: The path for the tokenizer model used
    
    */
    if(tokenizer_model.has_value()){
        model_path = tokenizer_model.value();
    }
    else{
        model_path = TOKENIZER_MODEL;
    }
    checkPath(model_path);
    const auto status = processor.Load(model_path);
    if(!status.ok()){
        std::cerr << status.ToString() << std::endl;
        return;
    }
    n_words = processor.GetPieceSize();
    bos_id = processor.bos_id();
    eos_id = processor.eos_id();
    pad_id = processor.pad_id();

}

void Tokenizer::checkPath(std::string& model_path){
    /*
    This method checks the model path given. Error thrown
    if model path is invalid

    @param model_path: The model path to the tokenizer provided
    */
    try{
        std::filesystem::path checkPath = model_path;
        if(std::filesystem::is_regular_file(checkPath)){
            std::cout << "Tokenizer model path exists";
        }
    }
    catch(const std::filesystem::filesystem_error& e){
        std::cout << "Error: " << e.what();
    }
}

std::vector<int> Tokenizer::encode(std::string s, bool bos, bool eos){
    std::vector<int> t = processor.EncodeAsIds(s);
    if(bos){
        t.push_back(bos_id);
    }
    if(eos){
        t.push_back(eos_id);
    }
    return t;
}

std::string Tokenizer::decode(std::vector<int> t){
    return processor.DecodeIds(t);
}

void Tokenizer::exporter(){
    std::vector<std::vector<uint8_t>> tokens;
    std::vector<float> scores;
    for(int i = 0; i < n_words; i++){
        std::string t = processor.IdToPiece(i);
        float s = processor.GetScore(i);
        if(i == bos_id){
            t = "\n<s>\n";
        }
        else if(i == eos_id){
            t = "\n</s>\n";
        }
        std::replace(t.begin(),t.end(),'_',' ');
        std::vector<uint8_t> b(t.begin(), t.end());

        tokens.push_back(b);
        scores.push_back(s);
    }
    uint32_t max_token_length = 0;
    for(std::vector<uint8_t> t : tokens){
        if(t.size() > max_token_length){
            max_token_length = t.size();
        }
    }
    std::string tokenizer_bin = std::regex_replace(model_path, std::regex(".model"), ".bin");
    std::ofstream outfile(tokenizer_bin, std::ios::out | std::ios::binary);
    if(!outfile){
        std::cerr << "Error: Could not open file for writing" << std::endl;
    }
    outfile.write(reinterpret_cast<const char*>(&max_token_length), sizeof(max_token_length));
    for(size_t i = 0; i < n_words; i++){
        float score = scores[i];
        uint32_t len = static_cast<uint32_t>(tokens[i].size());
        outfile.write(reinterpret_cast<const char*>(&score),sizeof(score));
        outfile.write(reinterpret_cast<const char*>(&len),sizeof(len));
        outfile.write(reinterpret_cast<const char*>(tokens[i].data()), len);
    }
}
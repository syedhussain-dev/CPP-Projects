#include "utils.h"
#include <cassert>
#include <filesystem>
#include <memory>
#include <stdexcept>

std::string saving_model(
    const std::shared_ptr<Transformer>& model,
    const std::string& model_name,
    const std::string& target_dir
){
    assert(
        (model_name.ends_with(".pth")) || (model_name.ends_with(".pt")) && "File must be a PyTorch Model"
    );
    std::filesystem::path model_path = target_dir;
    std::filesystem::create_directories(model_path);
    std::filesystem::path model_save_path = model_path / model_name;
    torch::save(model,model_save_path.string());
    return model_save_path.string();
}

void loading_model(
    std::shared_ptr<Transformer>& model,
    const std::string& model_save_path,
    const torch::Device& device
){
    if(!std::filesystem::is_regular_file(model_save_path)){
        throw std::runtime_error("File not Found!");
    }
    torch::load(model,model_save_path);
    model->eval();
}
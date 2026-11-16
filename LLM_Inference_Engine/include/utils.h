#ifndef UTILS_H
#define UTILS_H

#include <c10/core/Device.h>
#include <torch/torch.h>
#include "model.h"
#include <string>

std::string saving_model(
    const std::shared_ptr<Transformer>& model,
    const std::string& model_name,
    const std::string& target_dir
);

void loading_model(
    std::shared_ptr<Transformer>& model,
    const std::string& model_save_path,
    const torch::Device& device
);

#endif
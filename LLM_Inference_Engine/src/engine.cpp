#include "engine.h"
#include <chrono>

torch::Tensor train(
    torch::nn::Module& model,
    torch::Tensor train_dataloader,
    torch::Tensor val_dataloader,
    torch::Tensor test_dataloader,
    int epochs, int divisor,
    std::string device,
    torch::optim::AdamW& optimiser,
    torch::nn::CrossEntropyLoss& loss_fn,
    torch::optim::LRScheduler& scheduler,
    std::string model_name,
    std::string target_dir,
    int vocab_size,
    int accum_steps,
    double_t grad_clip
){
    auto start = std::chrono::high_resolution_clock::now();
    int patience = 20;
}

int detect_overfitting(
    std::map<std::string,std::vector<double_t>> results,
    int epoch,
    int& overfit_counter
){
    double_t min_delta = 0.01;
}

std::pair<double_t,int> stagnation(
    double_t current_loss,
    double_t best_loss,
    int epochs_no_imp
){

}
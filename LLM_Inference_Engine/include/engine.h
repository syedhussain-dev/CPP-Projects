#include <torch/optim/schedulers/lr_scheduler.h>
#include <torch/torch.h>
#include <map>
#include <string>
#include <utility>
#include "text_dataset.h"
#include "model.h"

using DatasetType = torch::data::datasets::MapDataset<TextDataset,torch::data::transforms::Stack<>>;
using DataLoaderType = torch::data::StatelessDataLoader<DatasetType,torch::data::samplers::RandomSampler>;

torch::Tensor train(
    Transformer& model,
    std::unique_ptr<DataLoaderType> train_dataloader,
    std::unique_ptr<DataLoaderType> val_dataloader,
    std::unique_ptr<DataLoaderType> test_dataloader,
    int epochs, int divisor,
    std::string device,
    torch::optim::AdamW& optimiser,
    torch::nn::CrossEntropyLoss& loss_fn,
    torch::optim::LRScheduler& scheduler,
    std::string model_name,
    std::string target_dir,
    int vocab_size,
    int accum_steps,
    double grad_clip=1.0
);

void detect_overfitting(
    std::map<std::string,std::vector<double>> results,
    int epoch,
    int& overfit_counter
);

std::pair<double,int> stagnation(
    double current_loss,
    double best_loss,
    int epochs_no_imp
);


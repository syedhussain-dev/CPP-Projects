#include "engine.h"
#include <c10/core/InferenceMode.h>
#include <iostream>
#include <chrono>
#include "utils.h"
#include <limits>
#include <torch/data/dataloader/stateless.h>
#include <torch/data/datasets/map.h>
#include <torch/data/samplers/random.h>
#include <torch/ordered_dict.h>

using DatasetType = torch::data::datasets::MapDataset<TextDataset,torch::data::transforms::Stack<>>;
using DataLoaderType = torch::data::StatelessDataLoader<DatasetType,torch::data::samplers::RandomSampler>;

torch::Tensor train(
    std::shared_ptr<Transformer>& model,
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
    double grad_clip
){
    auto start = std::chrono::high_resolution_clock::now();
    int patience = 20;
    int overfit_counter, epochs_no_imp = 0;
    double best_model_loss = std::numeric_limits<double>::infinity();
    double best_stagnation_loss = std::numeric_limits<double>::infinity();
    torch::OrderedDict<std::string,torch::Tensor> best_model_weights = model->named_parameters();
    std::map<std::string,std::vector<double>> results;
    std::vector<double> train_loss, val_loss;        
    size_t num_batches = 0.0;
    
    for(int epoch = 0; epoch < epochs; epoch++){
        auto epoch_start = std::chrono::high_resolution_clock::now();
        double train_loss_sum, val_loss_sum = 0.0;
        std::cout << "Epoch: " << epoch+1 << " -----------------------------------------------------------------";
        model->train(true);
        optimiser.zero_grad();

        for(auto& batch : *train_dataloader){
            torch::Tensor x = batch.data;
            torch::Tensor y = batch.target;
            auto train_logits = model->forward(x);
            auto loss = loss_fn(train_logits.view({-1,vocab_size}),y.view(-1));
            train_loss_sum += loss.item<double>();
            optimiser.zero_grad();
            loss.backward();
            optimiser.step();
            num_batches++;
        }
        train_loss.push_back(train_loss_sum / num_batches);
        num_batches = 0.0;
        model->eval();
        c10::InferenceMode guard;
        for(auto& batch : *val_dataloader){
            torch::Tensor x = batch.data;
            torch::Tensor y = batch.target;
            auto val_logits = model->forward(x);
            auto loss = loss_fn(val_logits.view({-1,vocab_size}),y.view(-1));
            val_loss_sum += loss.item<double>();
            num_batches++;
        }
        val_loss.push_back(val_loss_sum / num_batches);

        results["train_loss"] = train_loss;
        results["val_loss"] = val_loss;

        detect_overfitting(results, epoch, overfit_counter);
        if(val_loss[epoch] < best_model_loss){
            best_model_loss = val_loss[epoch];
            best_model_weights = model->named_parameters();
        }

        stagnation(val_loss[epoch],best_model_loss,epochs_no_imp);

        if(epochs_no_imp > patience){
            std::cout << "Model is no longer improving" << std::endl;
            break;
        }

        auto epoch_time_elapsed = std::chrono::high_resolution_clock::now() - epoch_start;
        if(epoch % divisor == 0){
            std::cout << "Train Loss: " << std::fixed << std::setprecision(5) << train_loss[epoch];
            std::cout << "Validation Loss: " << std::fixed << std::setprecision(5) << val_loss[epoch];
            std::cout << "Time Elapsed: " << std::fixed << std::setprecision(2) << epoch_time_elapsed;
        }

        scheduler.step();

    }
    for(const auto& param : model->named_parameters()){
        best_model_weights.insert(
            param.key(),
            param.value().detach().clone()
        );
    }
    model->eval();
    double test_loss_sum = 0.0;
    num_batches = 0.0;
    c10::InferenceMode guard;
    for(auto& batch : *test_dataloader){
        torch::Tensor x = batch.data;
        torch::Tensor y = batch.target;
        torch::Tensor test_logits = model->forward(x);
        test_loss_sum += (loss_fn(test_logits.view({-1,vocab_size}),y.view(-1))).item<double>();
    }
    auto time_elapsed = std::chrono::high_resolution_clock::now() - start;
    double test_loss = test_loss_sum / num_batches;
    std::cout << "Test Loss: " << std::fixed << std::setprecision(5) << test_loss;
    std::cout << "Total Training Time: " << time_elapsed;

    std::string model_save_path = saving_model(model,model_name,target_dir);
    std::cout << "Model saved to: " << model_save_path << std::endl;
    
}

void detect_overfitting(
    std::map<std::string,std::vector<double>>& results,
    int& epoch,
    int& overfit_counter
){
    if(epoch == 0) return;
    double_t min_delta = 0.01;

    std::vector<double> train_loss = results["train_loss"];
    std::vector<double> val_loss = results["val_loss"];

    if(val_loss[epoch]>val_loss[epoch-1] + min_delta &&
    train_loss[epoch]<train_loss[epoch-1 - min_delta]){
        overfit_counter = overfit_counter + 1;
    }
    else{
        overfit_counter = 0;
    }
    if(overfit_counter > 3){
        std::cout << "Model is overfitting" << std::endl;
    }

}

void stagnation(
    double& current_loss,
    double& best_loss,
    int& epochs_no_imp
){
    double min_delta = 1e-4;
    if(current_loss < best_loss - min_delta){
        best_loss = current_loss;
        epochs_no_imp = 0;
    }
    else{
        epochs_no_imp = epochs_no_imp + 1;
    }
}
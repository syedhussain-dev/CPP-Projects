#include "train.h"
#include "model.h"
#include "text_dataset.h"
#include "tokenizer.h"
#include "utils.h"
#include <string>
#include <torch/data/dataloader.h>
#include <torch/data/dataloader_options.h>
#include <torch/data/samplers/random.h>
#include <torch/data/transforms/stack.h>
#include <torch/types.h>
#include <vector>

void training(){
    const int64_t seq_len = 16;    
    Tokenizer tokenizer;
    std::string train_prompt = "Every morning before the rest of the street wakes up, Arthur unlocks the heavy wooden doors of his corner bakery and flips on the warm overhead lights. The air quickly fills with the sharp, comforting scent of yeast, toasted sourdough, and dark roast coffee brewing behind the counter. As the first commuters hurry past the steamed-up front window, he arranges trays of flaky, golden croissants onto the cooling racks, taking a quiet moment to enjoy the calm before the morning rush begins.";
    std::string val_prompt;
    std::string test_prompt;

    std::vector<int> train_tokens = tokenizer.encode(train_prompt, true, true);
    torch::Tensor train_token_tensor = torch::tensor(train_tokens,torch::kInt64);

    auto train_dataset = TextDataset(train_token_tensor, seq_len).map(torch::data::transforms::Stack<>());
    auto train_dataloader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(train_dataset),torch::data::DataLoaderOptions().batch_size(2));

}

void inferencing(){

}

int main(){
    return 0;
}
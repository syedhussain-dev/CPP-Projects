#include "text_dataset.h"

TextDataset::TextDataset(torch::Tensor tokens, int64_t seq_len){
    this->tokens = tokens;
    this->seq_len = seq_len;
}

torch::optional<size_t> TextDataset::size() const{
    if(seq_len <= 0 || tokens.numel() <= seq_len) return 0;
    return static_cast<size_t>(tokens.numel() - seq_len);
}

torch::data::Example<> TextDataset::get(size_t index){
    int64_t start_idx = static_cast<int64_t>(index);
    torch::Tensor x = tokens.slice(0,start_idx,start_idx+seq_len).to(torch::kInt64);
    torch::Tensor y = tokens.slice(0,start_idx+1,start_idx+seq_len+1).to(torch::kInt64);
    return {x,y};
}
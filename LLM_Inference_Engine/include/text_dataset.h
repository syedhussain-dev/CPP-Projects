#include <torch/data/datasets/base.h>
#include <torch/torch.h>

struct TextDataset : torch::data::datasets::Dataset<TextDataset>{

    TextDataset(torch::Tensor tokens, int64_t seq_len);

    torch::data::Example<> get(size_t index) override;

    torch::optional<size_t> size() const override;

    torch::Tensor tokens;
    int64_t seq_len;
};
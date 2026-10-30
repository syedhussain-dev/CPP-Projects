#include <ATen/ops/scaled_dot_product_attention.h>
#include <cstdint>
#include <torch/nn/modules/dropout.h>
#include <torch/nn/options/activation.h>
#include <torch/nn/options/embedding.h>
#include <torch/nn/options/linear.h>
#include <torch/torch.h>
#include <optional>
#include <cassert>
#include <vector>

class ModelArgs{
public:
    const int64_t dim = 4096;
    const int64_t num_encoder_layers = 32;
    const int64_t heads = 32;
    const std::optional<int64_t> num_kv_heads;
    const int64_t vocab_size = 32000;
    const std::optional<int64_t> hidden_dim;
    const int64_t multiple_of = 256;
    const double norm_eps = 1e-5;
    const int64_t max_seq_len = 2048;
    const double dropout = 0.0;
};

struct RMSNorm : torch::nn::Module{
    RMSNorm(int64_t dim, double eps);

    torch::Tensor _norm(torch::Tensor x);

    torch::Tensor forward(torch::Tensor x);

    torch::Tensor weight;
    double eps;
};

std::vector<torch::Tensor> precompute_freqs_cis(int64_t dim, int64_t end, double theta = 10000.0);

torch::Tensor reshape_for_broadcast(torch::Tensor freqs_cis, torch::Tensor x);

std::vector<torch::Tensor> RoPE(
    torch::Tensor xq, torch::Tensor xk,
    torch::Tensor freqs_cos, torch::Tensor freqs_sin
);

torch::Tensor repeat_kv(torch::Tensor x, int64_t n_rep);

struct Attention : torch::nn::Module{
    Attention(ModelArgs args);

    torch::Tensor forward(torch::Tensor x,torch::Tensor freqs_cos,torch::Tensor freqs_sin);

    int64_t num_kv_heads;
    int64_t num_local_heads;
    int64_t num_local_kv_heads;
    int64_t num_rep;
    int64_t head_dim;
    torch::nn::Linear wq;
    torch::nn::Linear wk;
    torch::nn::Linear wv;
    torch::nn::Linear wo;
    torch::nn::Dropout attn_dropout;
    torch::nn::Dropout resid_dropout;
    double dropout;
};

struct FeedForward : torch::nn::Module{
    FeedForward(int64_t dim, int64_t hidden_dim, int64_t multiple_of, double dropout);
    torch::Tensor forward(torch::Tensor x);
    torch::nn::Linear w1;
    torch::nn::Linear w2;
    torch::nn::Linear w3;
    torch::nn::Dropout dropout;
    torch::nn::SiLU silu;
};

struct TransformerBlock : torch::nn::Module{
    TransformerBlock(int64_t layer_id, ModelArgs args);
    torch::Tensor forward(torch::Tensor x, torch::Tensor freqs_cos, torch::Tensor freqs_sin);
    int64_t num_heads;
    int64_t dim;
    int64_t head_dim;
    int64_t layer_id;
    Attention attention;
    FeedForward feed_forward;
    RMSNorm ff_norm;
    RMSNorm attention_norm;

};

struct Transformer : torch::nn::Module{
    Transformer();
    torch::Tensor forward(torch::Tensor tokens, std::optional<torch::Tensor> targets=std::nullopt);
    ModelArgs params;
    int64_t vocab_size;
    int64_t num_encoder_layers;
    torch::nn::Embedding token_embeddings;
    torch::nn::Dropout dropout;
    torch::nn::ModuleList layers{nullptr};
    RMSNorm norm;
    torch::nn::Linear output;
    torch::Tensor freqs_cos;
    torch::Tensor freqs_sin;
};

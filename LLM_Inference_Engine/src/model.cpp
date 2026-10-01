#include <torch/torch.h>
#include <optional>
#include <list>
#include <cmath>

class ModelArgs{
public:
    int64_t dim = 4096;
    int64_t num_encoder_layers = 32;
    int64_t heads = 32;
    std::optional<int64_t> num_kv_heads;
    int64_t vocab_size = 32000;
    std::optional<int64_t> hidden_dim;
    int64_t multiple_of = 256;
    double_t norm_eps = 1e-5;
    int64_t max_seq_len = 2048;
    double_t dropout = 0.0;
};

struct RMSNorm : torch::nn::Module{
    RMSNorm(int64_t dim, double_t eps){
        weight = register_parameter("weight",torch::ones(dim));
        this->eps = eps;
    }

    torch::Tensor _norm(torch::Tensor x){
        return x * torch::rsqrt(torch::mean(torch::pow(x,2)) + eps);
    }

    torch::Tensor forward(torch::Tensor x){
        torch::Tensor output = _norm(x);
        return output;
    }

    torch::Tensor weight;
    double_t eps;
};

std::list<torch::Tensor> precompute_freqs_cis(int64_t dim, int64_t end, double_t theta = 10000.0){
    torch::Tensor freqs = 1.0 / torch::pow(theta, torch::arange(0,dim,2,torch::kFloat32) / dim);
}

struct Attention : torch::nn::Module{
    Attention(ModelArgs args){
        if(args.num_kv_heads.has_value()){
            num_kv_heads = args.num_kv_heads.value();
        }
        else num_kv_heads = args.heads;
        int64_t model_parallel_size = 1;
        num_local_heads = args.heads / model_parallel_size;
        num_local_kv_heads = num_kv_heads / model_parallel_size;
        num_rep = num_local_heads / num_local_kv_heads;
        head_dim = args.dim / args.heads;

        wq = torch::nn::Linear(torch::nn::LinearOptions(args.dim, args.heads * head_dim).bias(false));
        wk = torch::nn::Linear(torch::nn::LinearOptions(args.dim, num_kv_heads * head_dim).bias(false));
        wv = torch::nn::Linear(torch::nn::LinearOptions(args.dim, num_kv_heads * head_dim).bias(false));
        wo = torch::nn::Linear(torch::nn::LinearOptions(args.heads * head_dim, args.dim).bias(false));

        attn_dropout = torch::nn::Dropout(args.dropout);
        resid_dropout = torch::nn::Dropout(args.dropout);
        dropout = args.dropout;
    }

    torch::Tensor forward(torch::Tensor x){
        return x;
    }

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
    double_t dropout;
};

struct FeedForward : torch::nn::Module{
    FeedForward(int64_t dim, int64_t hidden_dim, int64_t multiple_of, double_t dropout){
        if(!hidden_dim){
            hidden_dim = 4 * dim;
            hidden_dim = static_cast<int>(2 * hidden_dim / 3);
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) / multiple_of);
        }
        w1 = torch::nn::Linear(torch::nn::LinearOptions(dim, hidden_dim).bias(false));
        w2 = torch::nn::Linear(torch::nn::LinearOptions(hidden_dim, dim).bias(false));
        w3 = torch::nn::Linear(torch::nn::LinearOptions(dim, hidden_dim).bias(false));
        this->dropout = torch::nn::Dropout(dropout);
    }
    torch::Tensor forward(torch::Tensor x){
        x = w1(x) * w3(x);
        x = silu(x);
        x = w2(x);
        return dropout(x);
    }
    torch::nn::Linear w1;
    torch::nn::Linear w2;
    torch::nn::Linear w3;
    torch::nn::Dropout dropout;
    torch::nn::SiLU silu;
};


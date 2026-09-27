#include <torch/torch.h>
#include <cmath>

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


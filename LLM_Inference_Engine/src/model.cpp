#include "model.h"

RMSNorm::RMSNorm(int64_t dim, double eps){
    weight = register_parameter("weight",torch::ones(dim));
    this->eps = eps;
}

torch::Tensor RMSNorm::_norm(torch::Tensor x){
    return x * torch::rsqrt(torch::mean(torch::pow(x,2)) + eps);
}

torch::Tensor RMSNorm::forward(torch::Tensor x){
    torch::Tensor output = _norm(x);
    return output;
}

std::vector<torch::Tensor> precompute_freqs_cis(int64_t dim, int64_t end, double theta){
    torch::Tensor freqs = 1.0 / torch::pow(theta, torch::arange(0,dim,2,torch::kFloat32) / dim);
    torch::Tensor t = torch::arange(end).to(freqs.device());
    freqs = torch::outer(t,freqs).to(torch::kFloat32);
    torch::Tensor freqs_cos = torch::cos(freqs);
    torch::Tensor freqs_sin = torch::sin(freqs);
    return {freqs_cos, freqs_sin};
}

torch::Tensor reshape_for_broadcast(torch::Tensor freqs_cis, torch::Tensor x){
    int64_t ndim = x.ndimension();
    std::vector<int64_t> shape;
    int64_t i = 0;
    assert(1 < ndim);
    assert(freqs_cis.size(0) == x.size(1) && freqs_cis.size(1) == x.size(-1));
    for(const auto& d : x.sizes()){
        if(i == 1 || i == ndim-1){
            shape.push_back(d);
        }
        else{
            shape.push_back(1);
        }
        i++;
    }
    freqs_cis = freqs_cis.view(shape);
    return freqs_cis;
}

std::vector<torch::Tensor> RoPE(
    torch::Tensor xq, torch::Tensor xk,
    torch::Tensor freqs_cos, torch::Tensor freqs_sin
){
    auto xq_parts = xq.to(torch::kFloat32).reshape({xq.size(0),xq.size(1),
        xq.size(2),-1,2}).unbind(-1);
    torch::Tensor xq_r = xq_parts[0];
    torch::Tensor xq_i = xq_parts[1];
    
    auto xk_parts = xk.to(torch::kFloat32).reshape({xk.size(0),xk.size(1),
        xk.size(2),-1,2}).unbind(-1);
    torch::Tensor xk_r = xk_parts[0];
    torch::Tensor xk_i = xk_parts[1];

    freqs_cos = reshape_for_broadcast(freqs_cos,xq_r);
    freqs_sin = reshape_for_broadcast(freqs_sin,xq_r);

    torch::Tensor xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin;
    torch::Tensor xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos;
    torch::Tensor xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin;
    torch::Tensor xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos;

    torch::Tensor xq_out = torch::stack({xq_out_r,xq_out_i},-1).flatten(3);
    torch::Tensor xk_out = torch::stack({xk_out_r,xk_out_i},-1).flatten(3);

    return {xq_out.type_as(xq),xk_out.type_as(xk)};

}

torch::Tensor repeat_kv(torch::Tensor x, int64_t n_rep){
    int64_t batch_size = x.size(0);
    int64_t seq_len = x.size(1);
    int64_t n_kv_heads = x.size(2);
    int64_t head_dim = x.size(3);

    if (n_rep == 1){
        return x;
    }

    return x.unsqueeze(3).expand({batch_size,seq_len,n_kv_heads,n_rep,head_dim}).reshape(
        {batch_size,seq_len,n_kv_heads*n_rep,head_dim}
    );
}


Attention::Attention(ModelArgs args){
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

torch::Tensor Attention::forward(torch::Tensor x,torch::Tensor freqs_cos,torch::Tensor freqs_sin){
    int64_t batch_size = x.size(0);
    int64_t seq_len = x.size(1);

    auto xq = wq(x);
    auto xk = wk(x);
    auto xv = wv(x);
    xq = xq.view({batch_size,seq_len,num_local_heads,head_dim});
    xk = xk.view({batch_size,seq_len,num_local_kv_heads,head_dim});
    xv = xv.view({batch_size,seq_len,num_local_kv_heads,head_dim});

    auto x_parts = RoPE(xq,xk,freqs_cos,freqs_sin);
    xq = x_parts[0];
    xk = x_parts[1];

    xk = repeat_kv(xk,num_rep);
    xv = repeat_kv(xv,num_rep);

    xq = xq.transpose(1,2);
    xk = xk.transpose(1,2);
    xv = xv.transpose(1,2);

    torch::Tensor output = at::scaled_dot_product_attention(
        xq, xk, xv, std::nullopt, dropout, true
    );

    output = output.transpose(1,2).contiguous().view({batch_size,seq_len,-1});
    output = wo(output);
    output = resid_dropout(output);
    return output;
}

FeedForward::FeedForward(int64_t dim, int64_t hidden_dim, int64_t multiple_of, double dropout){
    if(!hidden_dim){
        hidden_dim = 4 * dim;
        hidden_dim = static_cast<int64_t>(2 * hidden_dim / 3);
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) / multiple_of);
    }
    w1 = torch::nn::Linear(torch::nn::LinearOptions(dim, hidden_dim).bias(false));
    w2 = torch::nn::Linear(torch::nn::LinearOptions(hidden_dim, dim).bias(false));
    w3 = torch::nn::Linear(torch::nn::LinearOptions(dim, hidden_dim).bias(false));
    this->dropout = torch::nn::Dropout(dropout);
}
torch::Tensor FeedForward::forward(torch::Tensor x){
    x = w1(x) * w3(x);
    x = silu(x);
    x = w2(x);
    return dropout(x);
}


TransformerBlock::TransformerBlock(int64_t layer_id, ModelArgs args)
    :attention_norm(args.dim,args.norm_eps),
    ff_norm(args.dim,args.norm_eps),
    attention(args),
    feed_forward(args.dim,args.hidden_dim.value(),args.multiple_of,args.dropout)
{
    num_heads = args.heads;
    dim = args.dim;
    this->layer_id = layer_id;
}
torch::Tensor TransformerBlock::forward(torch::Tensor x, torch::Tensor freqs_cos, torch::Tensor freqs_sin){
    auto h = x + attention.forward(attention_norm.forward(x),freqs_cos,freqs_sin);
    auto out = h + feed_forward.forward(ff_norm.forward(h));
    return out;
}



Transformer::Transformer():norm(params.dim,params.norm_eps)
{
    vocab_size = params.vocab_size;
    num_encoder_layers = params.num_encoder_layers;
    token_embeddings = torch::nn::Embedding(torch::nn::EmbeddingOptions(params.vocab_size,params.dim));
    dropout = torch::nn::Dropout(params.dropout);
    layers = register_module("layers",torch::nn::ModuleList());
    for(int a = 0; a < num_encoder_layers; a++){
        layers->push_back(std::make_shared<TransformerBlock>(a,params));
    }
    output = torch::nn::Linear(torch::nn::LinearOptions(params.dim,params.vocab_size).bias(false));
    token_embeddings->weight = output->weight;

    std::vector<torch::Tensor> freqs = precompute_freqs_cis(params.dim/params.heads,params.max_seq_len);
    
    freqs_cos = register_buffer("freqs_cos",freqs[0]);
    freqs_sin = register_buffer("freqs_sin",freqs[1]);
    
}
torch::Tensor Transformer::forward(torch::Tensor tokens, std::optional<torch::Tensor> targets){
    int64_t batch_size = tokens.size(0);
    int64_t seq_len = tokens.size(1);
    torch::Tensor h = token_embeddings(tokens);
    h = dropout(h);

    torch::Tensor fc = freqs_cos.slice(0,0,seq_len);
    torch::Tensor fs = freqs_sin.slice(0,0,seq_len);

    for(auto& layer : *layers){
        auto block = layer->as<TransformerBlock>();
        h = block->forward(h,freqs_cos,freqs_sin);
    }
    h = norm.forward(h);
    torch::Tensor logits = output(h);
    return logits;
}


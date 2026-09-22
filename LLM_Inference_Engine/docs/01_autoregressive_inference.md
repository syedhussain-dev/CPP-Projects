# Autoregessive Inference
## What is it?
Generative process where a model predicts the next output using information
from the previous output for assistance. The newly generated token is then
added to the existing output. This tokens information is then used in the 
next prediction.

## Why does the inference engine use it?
Large Language models used this to provide responses to the user in a 
cohesive manner. 

## How this links to other concepts?
KV Caching. - More in the second section

## Tensor shapes
(Batch Size, Sequence Length, Hidden Dimension)

## How does the engine implement it?
There are two stages of implementation. The Prefill phase which is done on
the entire sequence of tokens during the attention step where the querys
of each token is matrix multiplied to the transpose of the key. To determine
attention scores and which information to exchange. Also, during the generation
phase where the model uses the previous tokens information in order to 
predict the next token.

## Concepts that I don't understand
I fully understand this concept.

## Sources
[1]L. Ting, “Spike-Aware C++ INT8 Inference for Sparse Spiking Language Models on Commodity CPUs.” June 02, 2026. Accessed: Sept. 21, 2026. [Online]. Available: https://arxiv.org/abs/2606.03026v1

## Images
No images provided.
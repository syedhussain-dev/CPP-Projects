# KV Cache
## ## What is it?
Tokens are converted converted to query, key and value representations.
The key and value tensors are stored within KV cache so the current token
can reuse the information from previous tokens.

## Why does the inference engine use it?
Without this, the model would need to recalculate the key and value from 
previous tokens which would require more computation and VRAM.

## How this links to other concepts?
This speeds up the process of autoregressive inference.

## Tensor shapes
Query: (Batch Size, Heads, Maximum Sequence Length, Head Dimension)
Key: (Batch Size, Heads, Maximum Sequence Length, Head Dimension)
Value: (Batch Size, Heads, Maximum Sequence Length, Head Dimension)

## How does the engine implement it?
Where autoregressive inference is used so during the prefill phase
and the generation phase.

## Concepts that I don't understand
I fully understnand this concept.

## Sources
[1]IBM Technology, “How KV Cache Speeds Up LLMs for Faster AI Models on GPUs.” YouTube, June 30, 2026. Accessed: Sept. 21, 2026. [Video]. Available: https://www.youtube.com/watch?v=o0gkdZBtwEg

## Images
No images provided.
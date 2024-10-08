# Efficient Streaming Lanaugage Models with Attention Sinks

## Problem:
2 major challenges with multi-round dialogue with LLMs

1. During the decoding stage, caching previous tokens' Key and Value states
consumes lots of memory. 
2. Popular LLMs cannot generalize to longer texts than the training sequence length.

Window attention, where only the most recent kv pairs are cached is a natural approach,
but this paper shows that it fails when text length surpasses cache size.

They found a phenomenon, an "attention sink", that keeping the KV of initial tokens
will largely recover the performance of window attention. 

Introduces the attention sink due to strong attention scores towards initial tokens,
and then introduce StreamingLLM, a framework that enables LLMs trained with a finite length
attention window to generalize to infinite sequence length without any fine tuning.

## Introduction

LLMs are contrained by the attention window during pre-training. There have been many efforts to expand the window size and improve training efficiency for lengthy inputs, but the acceptable sequence length remains instrisically finite. 

### What is LLM streaming?
Streaming enables LLMs trained with a finite attention window to work on text of 
infinite length without finetuning. It exploits the fact that attention sinks have 
high attention values, and preserving them can maintain attentions core distribution
close to normal. StreamingLLM keeps the atttention sink tokens' KV together with the 
sliding window's KV to anchor the attentionc omptuation and stabilize the model's performance.

### What has been done?
- Length Extrapolation
    - Researching how to enable language models trained on shorter texts to handle longer ones during testing
    - Rotary Position Embeddings (RoPE) - transforms keys and queries in every attention layer for relative position integration
    - ALiBi - biases query-key attention scores based on their distance, introducing relative positional information
    - Current methodologies have yet ot achieve infinite length extrapolation (no existing LLMs to fit for streaming applications)
- Context Window Extension
    - centered on expanding context window, enabling processing of more tokens in one forward pass
    - both a computational and memory challenge
- Improving LLM's Utilization of Long Text
    - optimizes LLMs to better capture and employ the content with context rather than merely taking them as inputs

## StreamingLLM
### What does it address?
- Window attention is unsuitable for deployment in streaming applications because there is a point of perplexity spike when the text length surpasses the cache size, led by the exclusion of initial tokens. This suggests that the initial tokens, regardless of their distance from the distance tokens, are crucial for maintaining the stability of the LLMs

### Why are the initial tokens so important?
- In figure 2, it is apparent that beyond the bottom 2 layers, the model consistently focuses on the initial tokens across all layers and heads
- Removing the initial tokens' KV will remove a considerable portion of the denominator in the SoftMax function, shifting distributino of attention scores away from what would be expected in normal inference settings. [INCLUDE EQUATION 1 HERE]
    - the paper hypothesized that either 1. the semantics are crucial or 2. the model learns a bias towards their absolute position
    - they conducted experiements [table 1] where the first four tokens are substituted with the linebreak token "\n"
        - the model still significantly emphasizes these initial linebreak tokens, and reintroducing them restores the language modeling perplexity to levels comparable to having the original initial tokens. This meant that the absolute position of the starting tokens, rather than their semantic value, holds greater significance

### LLMs attend to initial tokens as attention sinks
- they introduce the concept of an "attention sink", due to the nature of the SoftMax function
     - The softmax function prevents all tokens from having zero values
     - Because of the sequential nature of autoregressive language modeling, initial tokens are visible to all subsequent tokens, while later tokens are only visible to a limited set of subsequent tokens
- Introduction of 4 initial tokens as attention sinks in Figure 2 suffice to restore LLM's performance. They hypothesize that incorporating a stable learnable token at the start of all training samples coudl act as a committed attention sink, eliminiating the need for multiple initial tokens to ensure consistent streaming

### Rolling KV Cache with Attention Sinks
- to enable LLM streaming in already trained LLMs, they reintroduce a few starting tokens' KV in the attention computation
- KV Caching in StreamLLM can be divided into 2 parts
    - Attention sinks (four initial tokens) stabilizing the attention computation
    - rolling KV cache retains the most recent tokens
    




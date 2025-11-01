# Lecture 12: Transformer and LLM (Part I)

## Quick Reference

|Item|Reference|
|---|---|
| Slides | [View Slides](https://drive.google.com/drive/folders/1A3P6IBuS8wCzLlpdRiQBO9b1uoK3pnPf?usp=sharing)|
| Video | [EfficientML.ai Lecture 12 - Transformer and LLM (Part I)](https://www.youtube.com/watch?v=mR4u6ZaCYe4)  |
|Lab| -- |
|Professor|[Song Han](https://github.com/songhan)|

## Overview

The lecture moves from general model efficiency techniques to application-specific optimizations, focusing on the **Transformer architecture** and the evolution of **Large Language Models (LLMs)**.

***

### 1. Transformer Basics and Limitations of Previous Models

The lecture starts by comparing the **Transformer** with previous language models, **Recurrent Neural Networks (RNNs)** and **Long Short-Term Memory (LSTMs)**:

* **RNN/LSTMs Limitations [[06:49](http://www.youtube.com/watch?v=mR4u6ZaCYe4&t=409)]:** They process tokens sequentially, creating a **dependency** across time steps. This limits **parallelism** and causes the **vanishing gradient problem** over long sequences, making it hard to capture long-range dependencies (e.g., a word's meaning depending on something said far earlier in the sentence).
* **Convolutional Networks [[07:07](http://www.youtube.com/watch?v=mR4u6ZaCYe4&t=427)]:** While parallel, they have limited **receptive field** or context, which is insufficient for the non-local dependencies common in natural language.
* **The Transformer Solution [[09:59](http://www.youtube.com/watch?v=mR4u6ZaCYe4&t=599)]:** The Transformer architecture solves the dependency problem by using **Self-Attention**, allowing every token to look at every other token in the sequence simultaneously, maximizing parallelism and long-range context modeling.

***

### 2. The Core Transformer Building Blocks

#### A. Multi-Head Attention (MHA)

* **Token Embedding [[12:40](http://www.youtube.com/watch?v=mR4u6ZaCYe4&t=760)]:** Each word/token is mapped into a **continuous vector** (embedding) instead of sparse one-hot encoding.
* **Query-Key-Value (QKV) [[13:45](http://www.youtube.com/watch?v=mR4u6ZaCYe4&t=825)]:** The core idea of self-attention is to project the input embedding into three separate vectors: **Query (Q)**, **Key (K)**, and **Value (V)**. This design is analogous to a retrieval system (like a search engine).
    * **Attention Score:** Calculated as $\text{softmax}(\frac{Q K^T}{\sqrt{d_k}})$, where $\mathbf{Q} K^T$ measures the **similarity** between the Query of the current token and the Keys of all other tokens.
    * **Output:** The final output is the **weighted sum** of the $\mathbf{V}$alues, using the attention scores as weights.
* **Multi-Head [[19:28](http://www.youtube.com/watch?v=mR4u6ZaCYe4&t=1168)]:** Using multiple, independent attention "heads" allows the model to capture different types of relationships and semantics simultaneously.

#### B. Other Components

* **Feed-Forward Network (FFN) [[22:18](http://www.youtube.com/watch?v=mR4u6ZaCYe4&t=1338)]:** A simple two-layer Multi-Layer Perceptron (MLP) applied independently to each token's vector. Its main purpose is to introduce **token-wise non-linearity** for feature modeling. It is often an **inverted bottleneck** with a large hidden dimension (e.g., $4D$).
* **Layer Normalization (LayerNorm) [[24:57](http://www.youtube.com/watch?v=mR4u6ZaCYe4&t=1497)]:** Normalizes features **across the feature dimension** for each token, ensuring the mean is zero and variance is one. This improves **training stability**. Modern LLMs typically use **Pre-Norm** (normalization before the MHA/FFN block) over the original **Post-Norm** design.
* **Residual Connection [[24:08](http://www.youtube.com/watch?v=mR4u6ZaCYe4&t=1448)]:** The "**Add & Norm**" step adds the input of a sub-layer to its output, allowing gradients to flow easily and helping the training of very deep networks.

#### C. Positional Encoding (PE)

* **The Need:** Since self-attention treats the input as an **unordered set**, positional information must be injected.
* **Original Absolute PE [[29:08](http://www.youtube.com/watch?v=mR4u6ZaCYe4&t=1748)]:** Uses a fixed **sinusoidal function** (sine and cosine waves) to generate a unique encoding for each position, which is then **added** to the word embedding.
* **Relative Positional Encoding (RoPE) [[43:26](http://www.youtube.com/watch?v=mR4u6ZaCYe4&t=2606)]:** More advanced methods like **Rotary Positional Embedding (RoPE)** (used in Llama) inject relative position by **modifying the Q and K dot product** through a rotation matrix. This allows the model to **generalize to sequence lengths** longer than what it saw during training ("train short, test long") [[42:06](http://www.youtube.com/watch?v=mR4u6ZaCYe4&t=2526)].

***

### 3. Advanced LLM Design and Efficiency

The second half of the lecture covers architecture variants and critical memory optimizations.

#### A. Transformer Architecture Variants

* **Encoder-Decoder (T5) [[34:25](http://www.youtube.com/watch?v=mR4u6ZaCYe4&t=2065)]:** Used for sequence-to-sequence tasks like translation or summarization. The **encoder** sees the full input; the **decoder** generates the output token by token.
* **Encoder-Only (BERT) [[37:08](http://www.youtube.com/watch?v=mR4u6ZaCYe4&t=2228)]:** Uses a **full attention mask** (all tokens can see all other tokens). Trained using tasks like **Masked Language Modeling (MLM)** for classification and discrimination tasks.
* **Decoder-Only (GPT, Llama) [[39:13](http://www.youtube.com/watch?v=mR4u6ZaCYe4&t=2353)]:** Uses a **masked attention** (a token can only see previous tokens). Used for **generative** tasks (predicting the next word) and is the foundation for modern conversational LLMs.

#### B. KV Cache and Optimization

* **The Problem [[53:06](http://www.youtube.com/watch?v=mR4u6ZaCYe4&t=3186)]:** During auto-regressive inference (generating one token after another), the model needs the Keys (K) and Values (V) of all previously generated tokens. Recalculating them is wasteful.
* **KV Cache:** Stores the K and V vectors of previous tokens. The **cache size grows linearly with sequence length** and batch size, becoming the main memory bottleneck for long-context tasks (e.g., summarizing a 4K-token paper can require over 10 GB of VRAM just for the KV cache) [[01:00:21](http://www.youtube.com/watch?v=mR4u6ZaCYe4&t=3621)].
* **Optimizations to Reduce KV Cache Size:**
    * **Multi-Query Attention (MQA) [[01:02:54](http://www.youtube.com/watch?v=mR4u6ZaCYe4&t=3774)]:** Shares a single K and V matrix across all attention heads.
    * **Grouped-Query Attention (GQA) [[01:03:36](http://www.youtube.com/watch?v=mR4u6ZaCYe4&t=3816)]:** A middle ground, grouping several Query heads to share one K/V head. GQA (used in Llama 2) offers a **better balance between memory reduction and accuracy** than MQA [[01:05:24](http://www.youtube.com/watch?v=mR4u6ZaCYe4&t=3924)].

#### C. Gated Linear Units (GLU) for FFN

* **The FFN Upgrade [[01:06:19](http://www.youtube.com/watch?v=mR4u6ZaCYe4&t=3979)]:** Modern LLMs like Llama use **SwiGLU** (Swish Gated Linear Unit) instead of the simple ReLU-based FFN.
* **Mechanism:** SwiGLU replaces a single linear layer with two linear layers whose outputs are multiplied element-wise by a **gating mechanism** (three matrix multiplications in total). This design improves model quality and is now a popular design choice.

***

### 4. Scaling Laws and Popular LLMs

The lecture highlights the trend of model scaling:

* **Emergent Abilities [[01:08:40](http://www.youtube.com/watch?v=mR4u6ZaCYe4&t=4120)]:** Complex abilities (like advanced arithmetic or reasoning) only **emerge** once the model's compute and size exceed certain thresholds.
* **In-Context Learning (ICL) [[01:10:21](http://www.youtube.com/watch?v=mR4u6ZaCYe4&t=4221)]:** Large models (like GPT-3) gain the ability to learn tasks from **examples provided directly in the prompt** (Few-Shot Learning) without requiring traditional fine-tuning with gradients.
* **Key Models [[01:13:54](http://www.youtube.com/watch?v=mR4u6ZaCYe4&t=4434)]:** The lecture reviews popular LLM designs, including **OPT**, **Bloom**, and the **Llama/Llama 2** family, noting that the Llama models leverage RoPE and GQA for efficiency and strong performance. The **Chinchilla Scaling Law** [[01:18:43](http://www.youtube.com/watch?v=mR4u6ZaCYe4&t=4723)] emphasizes that not only parameter count but also the **amount of training data (tokens)** must scale proportionally to achieve optimal results.
## References

- EfficientML.ai Course | 2023 Fall | MIT 6.5940: [ Complete course video series ](https://youtube.com/playlist?list=PL80kAHvQbh-pT4lCkDT53zT8DKmhE0idB&si=Uu00N0zKopEixhw3).
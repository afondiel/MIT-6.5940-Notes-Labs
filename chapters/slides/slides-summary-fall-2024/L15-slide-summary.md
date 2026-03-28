# **Lesson 15: Long-Context LLM** 

focuses on techniques to extend the context window of Large Language Models, evaluate their performance over long sequences, and optimize attention mechanisms to reduce memory and latency bottlenecks.

### **I. Context Extension Techniques**
*   **Rotary Position Embedding (RoPE):** A method that rotates 2D pairs of embeddings based on their position to represent relative distance, which is foundational for models like Llama to "train short and test long".
*   **LongLoRA:** An efficient fine-tuning approach that makes context extension affordable by using **Shifted Sparse Attention**. It groups tokens and shifts them across attention heads during training, while also fine-tuning the input embedding and normalization layers.

### **II. Evaluation of Long-Context Abilities**
*   **Lost-in-the-Middle Phenomenon:** LLMs often exhibit a U-shaped performance curve where accuracy is high for information at the beginning or end of a document but drops significantly for information in the middle. 
*   **Needle In A Haystack (NIAH):** A diagnostic test that measures a model's ability to retrieve a specific fact (the "needle") placed at varying depths within a long document (the "haystack").
*   **LongBench:** A comprehensive benchmark consisting of 21 datasets across six task types (e.g., summarization, few-shot learning) to evaluate how well models handle real-world long-context tasks.

### **III. Efficient Attention Mechanisms**
*   **StreamingLLM:** Addresses the "model collapse" that occurs in window attention when initial tokens are evicted. It preserves **Attention Sinks** (the first few tokens) to stabilize the model, allowing it to handle infinite text lengths without additional training.
*   **DuoAttention:** Optimizes the KV cache by distinguishing between **Retrieval Heads**, which require full context to find earlier information, and **Streaming Heads**, which only need recent tokens and sinks. This framework reorders heads into clusters for efficient processing, reducing memory usage by up to 2.45x.
*   **Quest:** A technique utilizing **query-aware sparsity** to speed up self-attention by up to 7.03x by focusing on only the most critical tokens.

### **IV. Beyond Transformers**
*   **Mamba (State-Space Models):** To solve the quadratic complexity of Transformers, Mamba replaces attention with **Selective State Space Models (SSMs)**, which provide **linear-time complexity** for long sequences. It uses a **parallel scan** algorithm to accelerate training, making it mathematically similar to computing a prefix sum.
*   **Jamba:** A hybrid model that interleaves Transformer and Mamba layers with **Mixture-of-Experts (MoE)** to balance memory efficiency, high throughput, and high performance within a single GPU.

## References
- Source: https://notebooklm.google.com/notebook/39b7bd5c-37ef-451c-83dc-98b62ec8dc04
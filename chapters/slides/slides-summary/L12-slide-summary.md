# **Lesson 12: Transformer and LLM** 

focuses on the foundational architecture of Transformers, their design evolutions, the scaling of Large Language Models (LLMs), and advanced topics like multi-modal models.

### **I. Transformer Basics**
The Transformer architecture was introduced to overcome the limitations of **RNNs and LSTMs**, which struggled with long-term relationships and limited training parallelism due to their sequential nature.
*   **Key Components:** A standard Transformer block consists of **Multi-Head Attention (MHA)** to model token relationships, a **Feed-Forward Network (FFN)** for feature modeling, **LayerNorm**, **residual connections**, and **positional encoding**.
*   **Tokenization & Embeddings:** Words are mapped to tokens and then into continuous word embeddings through look-up tables (e.g., Word2Vec).
*   **Positional Encoding (PE):** Since attention and FFNs do not inherently differentiate token order, unique positional information is fused into raw embeddings to provide sequence context.

### **II. Transformer Design Variants**
Recent improvements have optimized the original design for efficiency and performance:
*   **KV Cache Optimization:** To reduce memory usage during inference, variants like **Multi-Query Attention (MQA)** (using one KV head) and **Grouped-Query Attention (GQA)** (using a subset of KV heads) were developed. GQA is notably used in m   odels like **Llama 2** to match MHA accuracy while saving memory.
*   **Activation Functions:** Replacing vanilla FFNs with **Gated Linear Units (GLU)**, such as **SwiGLU**, has been shown to improve Transformer performance.

### **III. Large Language Models (LLMs)**
LLMs are scaled-up Transformers trained on massive datasets.
*   **Emergent Abilities:** Scaling unlocks capabilities not found in smaller models, such as **modified arithmetic** and **word unscrambling**.
*   **In-Context Learning:** Large models like **GPT-3** act as **few-shot learners**, generalizing to new tasks through task descriptions (**zero-shot**) or a few demonstrations (**few-shot**) without requiring weight updates.
*   **Scaling Laws:** The **Chinchilla Law** suggests scaling both model and data size for optimal training, though for inference efficiency, it is often better to train smaller models (like **LLaMA**) on more tokens than traditionally recommended.

### **IV. Advanced Topics**
*   **Multi-modal LLMs:** New architectures enable models to process visual inputs. Approaches include **Flamingo**, which uses cross-attention to inject visual info into a frozen LLM, and **PaLM-E/VILA**, which treats visual information as input tokens.
*   **Mixture-of-Experts (MoE):** This technique uses **sparse activation**, where only a subset of "experts" is triggered for each token, controlled by a routing function to manage computational costs.
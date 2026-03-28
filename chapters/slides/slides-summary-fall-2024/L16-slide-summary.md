# **Lesson 16: Vision Transformer** 

focuses on the adaptation of the Transformer architecture for computer vision tasks, techniques for improving its efficiency, self-supervised learning methods, and its application in autoregressive image generation.

### **I. Basics of Vision Transformer (ViT)**
Unlike CNNs that use sliding windows, **ViT treats an image as a sequence of patches**, similar to tokens in a sentence.
*   **Patching and Projection:** A 2D image is divided into fixed-size patches (e.g., 32x32), which are flattened and mapped into a sequence of embeddings through a **linear projection**. 
*   **Standard Transformer Encoder:** These patch embeddings, fused with **positional embeddings**, are fed into a standard Transformer encoder consisting of Multi-Head Attention (MHA) and Feed-Forward Networks (FFN).
*   **Scaling and Performance:** ViT variants range from **Base (12 layers)** to **Huge (32 layers)**. While ViT performance is often inferior to CNNs on small datasets, it **surpasses CNNs when pre-trained on massive datasets** like JFT-300M.

### **II. Efficient ViT and Acceleration**
To handle high resolutions and reduce the $O(n^2)$ complexity of standard attention, several efficiency techniques are introduced:
*   **Linear Attention:** Replaces traditional Softmax attention with **linear attention** to reduce computational cost.
*   **SparseViT:** This framework uses **window activation pruning** to achieve non-uniform sparsity. It involves a three-step process: pruning windows based on importance, performing sparsity-aware adaptation during fine-tuning, and using an evolutionary search to find the optimal configuration under a latency budget.
*   **Results:** SparseViT effectively **prunes irrelevant background windows** while retaining foreground information, achieving up to 1.5x speedups in tasks like 3D object detection.

### **III. Self-Supervised Learning (SSL)**
Since labeling massive datasets is expensive, SSL is used to train ViT on unlabeled data.
*   **Pretext Tasks:** Common strategies include **contrastive learning** and **masked image modeling**, where the model learns to reconstruct missing parts of an image or distinguish between different views of the same data.

### **IV. Hybrid Autoregressive Transformer (HART)**
The lesson introduces **HART**, a novel approach for efficient visual generation that combines autoregressive (AR) modeling with residual diffusion.
*   **Hybrid Tokenization:** HART uses a hybrid tokenizer to decompose an image into **discrete tokens** (representing the "big picture") and **residual tokens** (representing fine details like hair and eyes).
*   **Scalable Architecture:** Discrete tokens are modeled by a scalable-resolution AR transformer using **relative position embeddings**, while the fine details are predicted by a **residual diffusion** model.
*   **Efficiency and Quality:** HART provides a **4.5x to 7.7x higher throughput** than traditional diffusion models (like SD-XL) while maintaining comparable visual quality. It is efficient enough to run at **interactive speeds on a laptop** equipped with a mobile GPU.

## References
- Source: https://notebooklm.google.com/notebook/39b7bd5c-37ef-451c-83dc-98b62ec8dc04
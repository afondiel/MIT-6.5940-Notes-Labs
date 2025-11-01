# Lecture 14: Vision Transformer (ViT) Efficiency

## Quick Reference

|Item|Reference|
|---|---|
| Slides | [View Slides](https://drive.google.com/drive/folders/1A3P6IBuS8wCzLlpdRiQBO9b1uoK3pnPf?usp=sharing)|
| Video | [EfficientML.ai Lecture 14 ](http://www.youtube.com/watch?v=6cAmS-_vEh8)  |
|Lab| -- |
|Professor|[Song Han](https://github.com/songhan)|


## 1. 🎯 Why It Matters for Edge AI

* **The Core Problem:** While Vision Transformers (ViTs) and their variants (Swin, DeiT) achieve state-of-the-art accuracy, they inherit the **quadratic complexity ($O(N^2)$) of self-attention** from LLMs. In vision, $N$ is the number of image patches, which can be very large, making vanilla ViT computationally expensive and memory-hungry compared to highly optimized Convolutional Neural Networks (CNNs).
* **Edge AI Benefits:** Developing efficient ViTs allows edge devices to leverage the superior accuracy and robustness of the Transformer architecture for complex vision tasks (like high-fidelity image recognition, video analysis, and object detection) where CNNs might struggle, all while meeting tight latency and power budgets.

---

## 2. 📝 Key Concepts and Theory

* **ViT Architecture Recap:**
    1.  **Patch Embedding:** The input image is split into fixed-size, non-overlapping **patches**. Each patch is flattened and linearly projected into a sequence of tokens.
    2.  **Transformer Encoder:** This sequence of tokens (patches) is processed by a standard Transformer Encoder stack, relying heavily on **Self-Attention** to capture global relationships between patches.
    3.  **Classification Head:** A standard MLP is used on the output token (usually a special **[CLS]** token) for final prediction.
* **The Efficiency Bottleneck:**
    * **High $N$:** For an image of size $224 \times 224$ with $16 \times 16$ patches, the sequence length $N$ is $196$. For a high-resolution image ($512 \times 512$), $N$ is much higher. The $O(N^2)$ attention calculation becomes a massive bottleneck.
    * **Memory Access:** The attention matrix and the resulting intermediate tensors can be very large, stressing the memory bandwidth on edge devices.
* **Structural Efficiency Improvements:**
    * **Hierarchical ViTs (e.g., Swin Transformer):** Introduce a hierarchical structure (similar to CNNs) where attention is calculated only within small **local windows** and then shifted across windows. This successfully reduces the complexity from $O(N^2)$ to **$O(N)$** complexity, making them much faster and more suitable for dense prediction tasks (segmentation, detection).
    * **MobileViT:** Attempts to fuse the benefits of local CNNs (for feature extraction) and global Transformers (for information aggregation) to build a mobile-friendly architecture.

---

## 3. ⚙️ Practical Implementation & Tools

* **Pruning and Quantization for ViT:** These techniques are applied to ViTs with some specific considerations:
    * **Pruning:** Pruning often targets the **MLP blocks** and the **Attention Heads**, as they tend to be highly redundant.
    * **Quantization:** Aggressive quantization (INT8) is used on ViTs. Due to the high dynamic range in the attention matrix (Softmax output), quantization is often applied with careful range clipping or using specialized methods like **Layer-wise Quantization-Aware Training (QAT)**.
* **Efficient Inference Kernels:** Similar to LLMs, fast ViT inference relies on specialized kernels:
    * **Flash Attention:** Used to speed up the core attention operation by optimizing memory movement on GPUs/NPUs.
    * **Optimized MLP Kernels:** Since the ViT is essentially a sequence of large matrix multiplications (linear layers and MLPs), deploying highly optimized matrix multiplication kernels (GEMM) is critical.

---

## 4. ⚖️ Trade-offs and Real-World Impact

* **Trade-off: CNN vs. ViT:** While efficient ViT variants (Swin, MobileViT) are competitive with modern CNNs (like MobileNetV3), the inherent complexity of ViTs means that CNNs often still hold a slight edge in terms of **absolute latency** and energy efficiency on the most constrained devices.
* **ViT Advantage:** ViTs tend to have **better transferability** (pre-trained ViTs generalize better to new tasks) and superior performance on **large-scale datasets**, justifying the efficiency investment.
* **The Patch Embedding Overhead:** The initial step of splitting the image into patches and embedding them requires specialized, fast operations and careful memory management.

---

## 5. 🧪 Hands-on Lab Preview

* **What you will do:** Compare the computational overhead of a vanilla ViT architecture against a hierarchical variant (like a toy Swin Transformer) by analyzing the FLOPs/MACs count for the attention block as the image resolution increases. You will also apply **Quantization** to a pre-trained ViT and measure its accuracy drop.
* **Key Skill Acquired:** Analyzing the impact of attention complexity in the vision domain, selecting the correct efficient ViT variant for a given task, and applying compression techniques tailored for the Transformer architecture.


## References

- EfficientML.ai Course | 2023 Fall | MIT 6.5940: [ Complete course video series ](https://youtube.com/playlist?list=PL80kAHvQbh-pT4lCkDT53zT8DKmhE0idB&si=Uu00N0zKopEixhw3).
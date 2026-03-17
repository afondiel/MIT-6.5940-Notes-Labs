# **Lesson 6: Quantization (Part II)** 

expands on the techniques for deploying low-precision neural networks, specifically focusing on Post-Training Quantization (PTQ), Quantization-Aware Training (QAT), ultra-low precision (Binary/Ternary), and automated mixed-precision strategies.

### **I. Post-Training Quantization (PTQ)**
PTQ quantizes a pre-trained floating-point model without requiring extensive retraining. The lecture focuses on three main optimization topics:

*   **Quantization Granularity:** 
    *   **Per-Tensor:** Uses a single scale for the entire weight tensor. While effective for large models, accuracy often drops for smaller models (e.g., MobileNetV2) due to "outlier" weights.
    *   **Per-Channel:** Assigns a unique scale to each output channel, which is crucial for handling large range differences between weights.
    *   **Group Quantization:** A finer granularity (e.g., **VS-Quant**) that uses hierarchical scaling factors (floating-point for coarse-grained, integer for fine-grained) to balance accuracy and hardware efficiency. This is supported by modern hardware like **NVIDIA Blackwell** for FP4 AI.
*   **Dynamic Range Clipping:** To determine optimal linear parameters ($S, Z$), statistics are gathered from activations. Methods include **Exponential Moving Averages (EMA)**, minimizing **KL Divergence**, or minimizing **Mean-Square-Error (MSE)** via algorithms like OCTAV.
*   **Rounding:** Simple "round-to-nearest" is often sub-optimal because weights are correlated. **AdaRound** is an adaptive method that finds the optimal rounding by minimizing the reconstruction error of the original activations.

### **II. Quantization-Aware Training (QAT)**
QAT emulates inference-time quantization during the training or fine-tuning process to recover lost accuracy.
*   **Simulated/Fake Quantization:** During the forward pass, weights and activations are quantized to ensure the model "sees" the discrete values it will use during inference.
*   **Straight-Through Estimator (STE):** Since the derivative of the rounding/quantization function is zero almost everywhere, the STE is used to pass gradients through as if the quantization function were the identity function, allowing weight updates.

### **III. Binary and Ternary Quantization**
These push quantization to the extreme limits to achieve massive storage and computation savings.
*   **Binary Quantization (1-bit):** Weights are constrained to $+1$ and $-1$. **XNOR-Net** allows the replacement of standard multiplications with **XNOR and popcount** operations, which can be up to 58x faster on supported hardware.
*   **Ternary Quantization (2-bit):** Weights are restricted to $\{+1, 0, -1\}$. **Trained Ternary Quantization (TTQ)** introduces trainable scaling factors for positive and negative weights, significantly improving accuracy over fixed-scale binarization.

### **IV. Automatic Mixed-Precision Quantization**
Instead of using a uniform bit-width for all layers, mixed-precision assigns different bit-widths to different layers based on their sensitivity.
*   **HAQ (Hardware-Aware Quantization):** This framework uses **Reinforcement Learning** (DDPG agent) to automatically search the massive design space of bit-widths. It provides direct hardware feedback to optimize models for specific objectives like minimum latency or energy consumption on edge and cloud devices.

## References
- Source: https://notebooklm.google.com/notebook/39b7bd5c-37ef-451c-83dc-98b62ec8dc04
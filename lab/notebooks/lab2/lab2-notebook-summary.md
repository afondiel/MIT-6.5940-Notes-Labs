# **Notebook Summary**

### Section 1: Setup and Baseline
*   **Environment Setup**: Installs `torchprofile` for MACs calculation and `fast-pytorch-kmeans` for clustering.
*   **Model Definition**: Defines a VGG-like architecture optimized for CIFAR-10.
*   **FP32 Baseline**: Loads a pretrained model and evaluates it, establishing a baseline accuracy of ~92.95% and a size of 35.2 MiB.

### Section 2: K-Means Quantization
*   **Clustering**: Implements `k_means_quantize` which clusters weights into $2^n$ centroids. This allows representing weights as indices, significantly reducing storage.
*   **Quantization-Aware Training (QAT)**: Implements `update_codebook` to refine centroids during training, which helps recover accuracy lost during the initial quantization, especially at 2-bit and 4-bit widths.

### Section 3: Linear Quantization
*   **Mechanism**: Implements $r = S(q - Z)$ mapping. It includes logic for calculating scale $S$ and zero-point $Z$.
*   **Fused Inference**: Fuses BatchNorm into Convolution layers to simplify the graph. It uses custom `QuantizedConv2d` and `QuantizedLinear` modules to simulate integer-only arithmetic.
*   **Post-Training Quantization (PTQ)**: Calibrates the model using sample data to determine activation ranges, achieving 92.87% accuracy with INT8.

## Comparison of K-Means-based Quantization and Linear Quantization based on Lab Results

Based on the lab, we can compare k-means-based quantization and linear quantization across several aspects:

### 1. K-Means Quantization

**Results from the Lab:**

*   **FP32 Model Baseline:**
    *   Accuracy: 92.95%
    *   Size: 35.20 MiB
*   **K-Means Quantization (without fine-tuning):**
    *   **8-bit:**
        *   Size: 8.80 MiB (4x reduction from FP32)
        *   Accuracy: 92.73% (0.22% drop from FP32)
    *   **4-bit:**
        *   Size: 4.40 MiB (8x reduction from FP32)
        *   Accuracy: 81.61% (11.34% drop from FP32)
    *   **2-bit:**
        *   Size: 2.20 MiB (16x reduction from FP32)
        *   Accuracy: 16.23% (76.72% drop from FP32)
*   **K-Means Quantization (after quantization-aware training/fine-tuning):**
    *   **8-bit:** No fine-tuning needed as accuracy drop was below threshold. Final accuracy: 92.73%.
    *   **4-bit:** Accuracy recovered to 92.70% (after 1 epoch).
    *   **2-bit:** Accuracy recovered to 91.47% (after 4 epochs).

**Advantages/Disadvantages (as per Question 10):**

*   **Advantages:**
    *   Easy to implement.
    *   Lower latency (implicitly, due to reduced bitwidth).
    *   Better hardware support (often easier to map to existing integer ALUs).
*   **Disadvantages:**
    *   Lower accuracy (can be significantly lower without fine-tuning, especially at very low bitwidths).

### 2. Linear Quantization

**Results from the Lab:**

*   **INT8 Model (post-training quantization):**
    *   Accuracy: 92.87% (only a 0.08% drop from FP32 baseline of 92.95%).
    *   Model size is not explicitly calculated as a variable but would be significantly reduced (e.g., to 1/4 of FP32 for weights, assuming 8-bit).

**Advantages/Disadvantages (as per Question 10):**

*   **Advantages:**
    *   Higher accuracy (often retaining near FP32 accuracy, as demonstrated by the 0.08% drop for INT8).
*   **Disadvantages:**
    *   Hard to implement (involves careful calculation of scales, zero points, and handling of intermediate integer operations).
    *   Higher latency (compared to simpler k-means inference, as it involves scaling and zero-point adjustments, although still faster than FP32).
    *   Needs extra hardware support for integer operations, e.g., for specific handling of maxpooling and avgpooling (as seen in the `QuantizedMaxPool2d` and `QuantizedAvgPool2d` implementations which temporarily convert to `float()`).

### Overall Comparison

| Feature           | K-Means Quantization                                 | Linear Quantization                                        |
| :---------------- | :--------------------------------------------------- | :--------------------------------------------------------- |
| **Accuracy**      | Significant drop at lower bitwidths without fine-tuning; recoverable with fine-tuning. | Achieves very high accuracy, close to FP32, even with post-training quantization (as shown by INT8). |
| **Model Size**    | Achieves significant reduction (e.g., 2.20 MiB for 2-bit, 4.40 MiB for 4-bit) | Achieves significant reduction (e.g., INT8 reduces to ~1/4 of FP32 size). |
| **Latency**       | Lower (simpler inference due to shared centroids).   | Higher than k-means due to scale/zero-point operations, but still faster than FP32. |
| **Implementation** | Easier to implement.                                 | More complex, requiring careful handling of scales, zero points, and integer arithmetic throughout the model. |
| **Hardware Support** | Generally better support, as it maps to simpler integer operations. | Requires specific hardware support for integer-only inference, especially for non-linear operations like pooling. |
| **Training**      | Quantization-aware training is often crucial to recover accuracy, especially at low bitwidths. | Can achieve good accuracy with post-training quantization, but quantization-aware training can further improve results for more challenging cases. |
| **Key Mechanism** | Clustering weights into a codebook, sharing centroid values. | Mapping floating-point ranges to integer ranges using a scale and zero-point. |

In summary, for achieving extremely high compression ratios (e.g., 2-bit or 4-bit), k-means quantization might be considered, though it heavily relies on quantization-aware training to maintain acceptable accuracy. Linear quantization, particularly at 8-bit, offers a good balance between model size reduction and accuracy preservation, often without extensive retraining, but comes with increased implementation complexity and specific hardware requirements for true integer-only inference.

## Final Notebook Summary

### Performance Comparison
| Metric | FP32 Baseline | K-Means (4-bit + QAT) | Linear (INT8) |
| :--- | :--- | :--- | :--- |
| **Accuracy** | 92.95% | ~92.70% | 92.87% |
| **Model Size** | 35.20 MiB | 4.40 MiB | ~8.80 MiB |
| **Reduction** | 1x | 8x | 4x |

### Key Takeaways
1.  **K-Means Quantization** is excellent for aggressive compression (up to 16x at 2-bit), but necessitates **Quantization-Aware Training** to maintain usability.
2.  **Linear Quantization (INT8)** provides a more standard deployment path, offering near-lossless accuracy (0.08% drop) without needing full retraining, provided that BatchNorm fusion and proper calibration are performed.
3.  **Implementation Trade-offs**: While K-Means is conceptually simpler, Linear Quantization is more hardware-friendly for standard integer units (ALUs) found in modern mobile and edge CPUs.
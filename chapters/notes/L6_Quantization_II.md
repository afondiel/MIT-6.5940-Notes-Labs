# Lecture 6: Quantization (Part II)

## Quick Reference

|Item|Reference|
|---|---|
| Slides | [View Slides](https://drive.google.com/drive/folders/1A3P6IBuS8wCzLlpdRiQBO9b1uoK3pnPf?usp=sharing)|
| Video | [EfficientML.ai Lecture 6 - Quantization (Part II)](https://www.youtube.com/watch?v=n72ndSimkB8)  |
|Lab| [Lab2.ipynb](../../lab/notebooks/Lab2.ipynb) |
|Professor|Song Han|


## **1\. Challenges with Low-Bit Quantization (INT4 and Below)**

While INT8 quantization (L5) is often straightforward, pushing precision down to 4-bit (INT4) or 2-bit (INT2) for extreme compression introduces major challenges, particularly with **outliers**.

### **A. The Outlier Problem**

* **Observation:** In large, modern neural networks (especially LLMs), the distribution of weights and activations contains a small number of **extreme outliers** (values far from the mean).  
* **Impact on Quantization:** When using a fixed, narrow bit-width (like INT4), the scaling factor ($S$) must be calculated to cover the entire range, including the outliers. This makes the steps between quantized values very large, leading to significant **rounding error** for the vast majority of non-outlier weights/activations.  
  * **Analogy:** It's like trying to measure objects from 1 to 100, but one object is at 1,000,000. If you set your scale to cover 1,000,000, your measurement for 10 is almost meaningless.  
* **Result:** A massive drop in model accuracy and breakdown of performance when quantizing models like **LLaMA** to INT4 using simple min-max techniques.

### **B. Groupwise vs. Layerwise Quantization**

To mitigate the outlier problem, the **quantization range** should be applied to smaller, more localized groups of data.

* **Layerwise Quantization (Simple PTQ):** One scale factor $S$ and zero point $Z$ are used for **all** weights/activations in an entire layer. (Prone to outlier problems).  
* **Groupwise Quantization:** A separate scale and zero point ($S$ and $Z$) are calculated for a small group of weights (e.g., a group of 128 weights). This allows the scale to adapt to local distributions, minimizing the impact of outliers in other parts of the layer.  
  * This technique is crucial for achieving high accuracy with INT4 weights.

## **2\. Advanced Quantization Algorithms for LLMs**

These techniques specifically target the quantization of large models like Transformers by mitigating the outlier problem in either weights or activations.

### **A. Activation-Aware Weight Quantization (AWQ)**

AWQ is an efficient post-training quantization (PTQ) method that tackles the sensitivity of LLMs to weight outliers.

* **Core Insight:** The importance of a weight is determined not just by its magnitude, but by the magnitude of the **activation** it interacts with. A weight with a moderate value, when multiplied by a massive activation outlier, has a huge impact on the output.  
* **Method:** AWQ searches for an optimal **channel-wise scaling factor** ($\\alpha$) for the weights *before* quantization. This factor is chosen to protect the weights that interact with large activation outliers.  
  * It uses a small calibration set to profile the distribution of activations.  
  * The protected weights are effectively kept in higher precision (or given a better quantization range) while the rest are quantized to INT4.  
* **Benefit:** AWQ achieves state-of-the-art INT4 weight quantization accuracy for LLMs with minimal overhead, making it widely adopted (e.g., in Hugging Face's AutoGPTQ library).

### **B. SmoothQuant**

SmoothQuant focuses on solving the high sensitivity to **activation outliers** in LLMs.

* **Core Insight:** Activations often have far more extreme outliers than weights, making activation quantization difficult. However, weights are less sensitive to quantization than activations.  
* **Method:** SmoothQuant **"smooths"** the highly dynamic activation distribution by moving the difficulty (outliers) from the activations to the weights.  
  * It applies a per-channel smoothing factor ($s$) to the activations.  
  * To maintain mathematical equivalence, the inverse factor ($1/s$) is then applied to the weights of the *next* layer.
  $$
  \text{Output} = \text{Layer2}(\text{Layer1}(\mathbf{X})) = \text{Layer2}(\mathbf{W}_1 \mathbf{X})
  $$
  
  $$
  \text{Layer2}(\mathbf{W}_1 \mathbf{X}) = \text{Layer2}((\mathbf{W}_1 \cdot \mathbf{s}) \cdot (\mathbf{s}^{-1} \mathbf{X}))
  $$  
  
  * The smoothed activations are now easily quantized, and the slightly modified weights can be quantized without severe loss.  
* **Benefit:** Allows the use of INT8 quantization for both weights and activations (W8A8) across the entire LLM, maintaining near FP16 accuracy.

## **3\. Deployment: Quantizing the KV Cache**

A major bottleneck for LLM inference latency and memory is the **Key-Value (KV) Cache**.

* **KV Cache Role:** During autoregressive decoding, the **Key** and **Value** vectors of previous tokens from the self-attention mechanism are stored in memory so they don't need to be recomputed for every new token.  
* **Memory Cost:** The KV cache memory requirement scales linearly with the sequence length and the batch size. For long sequences and large models, the KV cache can dominate VRAM usage.  
* **Quantizing the KV Cache:** Quantizing the Key and Value vectors (often to INT8) is a crucial technique for LLM deployment:  
  * It effectively **reduces the LLM's memory consumption** (VRAM usage) by 2x (for INT8) or 4x (for INT4), allowing for longer context windows or larger batch sizes.  
  * This is a form of **activation quantization** specific to the Transformer architecture.

## **4\. Summary of Quantization Strategies**

| Strategy | Goal | Complexity | Best Accuracy | Common Target |  
|----------------------|----------------------|----------------------|----------------------|----------------------|
| PTQ (Min-Max) | Quick compression | Low | Good (INT8) | Vision Models, Simple INT8 |  
| QAT | Max accuracy at low-bit | High | Best (INT2/INT4) | Extreme edge, maximum accuracy required |  
| AWQ | INT4 Weight Quant. | Medium | State-of-Art INT4 W | LLMs, reducing weight memory |  
| SmoothQuant | INT8 Activation Quant. | Medium | High INT8 A | LLMs, reducing activation memory |

## References

- EfficientML.ai Course | 2023 Fall | MIT 6.5940: [ Complete course video series ](https://youtube.com/playlist?list=PL80kAHvQbh-pT4lCkDT53zT8DKmhE0idB&si=Uu00N0zKopEixhw3).

## Additional Resources
- [Quantization in Depth - A Crash Course by Hugging Face (DeepLearning.ai)](https://github.com/afondiel/Quantization-in-Depth-HF)
- [Quantization Fundamentals with Hugging Face (DeepLearning.ai Course)](https://github.com/afondiel/Quantization-Fundamentals-with-HF)
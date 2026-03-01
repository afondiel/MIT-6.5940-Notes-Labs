# Lecture 5: Quantization (Part I)

## Quick Reference

|Item|Reference|
|---|---|
| Slides | [Slides](https://drive.google.com/drive/folders/1A3P6IBuS8wCzLlpdRiQBO9b1uoK3pnPf?usp=sharing)|
| Video | [EfficientML.ai Lecture 5 - Quantization (Part I)](https://www.youtube.com/watch?v=MK4k64vY3xo)  |
|Lab| [Lab2-Quantization.ipynb](../../lab/notebooks/Lab2.ipynb) |
|Professor|[Song Han](https://github.com/songhan)|


## **1\. Introduction and Motivation for Quantization**

Quantization is the process of reducing the precision of the numerical representations in a neural network, typically moving from 32-bit floating-point (FP32) to lower-bit formats like 8-bit integer (INT8) or even lower.

### **A. Why Quantize?**

The primary benefits of quantization directly address the AI Computing Gap discussed in [L1](./L01_Introduction.md):

1. **Reduced Model Size:** An INT8 weight takes up $\\frac{1}{4}$ the memory of an FP32 weight. This is critical for memory-constrained devices (TinyML, mobile).  
2. **Faster Inference:** Integer arithmetic operations (INT8) are significantly faster and require less complex hardware than floating-point operations (FP32).  
3. **Lower Power Consumption:** Reduced data movement and simpler arithmetic lead to massive energy savings, crucial for battery-powered edge devices.  
4. **Hardware Compatibility:** Quantization enables deployment on specialized low-power hardware (ASICs, microcontrollers, specialized Tensor Cores) optimized for integer math.

### **B. Common Data Types in ML**

| Data Type | Precision (Bits) | Size (Bytes) | Typical Use Case |
| :---- | :---- | :---- | :---- |
| **FP32** (Float) | 32 | 4 | Training, High-Precision Inference (Default) |
| **FP16** (Half-Precision) | 16 | 2 | Training large models, GPU Inference (Mixed Precision) |
| **BF16** (Brain Float) | 16 | 2 | Training large models (better range than FP16) |
| **INT8** (Integer) | 8 | 1 | **Most common Inference Quantization target** |
| **INT4 / INT2 / Binary** | 4, 2, 1 | $<1$ | Extreme Compression / TinyML |

## **2\. Quantization Fundamentals: Linear Quantization**

The most common method is **linear quantization**, which maps a range of real-valued floating-point numbers ($x_{f}$) to a limited set of discrete, fixed-point integer values ($x_{q}$).

### **A. The Mapping Equation**

The mapping is defined by a **Scale Factor (**$S$**)** and a **Zero Point (**$Z$**)**:

$$x_{q} = \text{Round}(\frac{x_{f}}{S} + Z)
$$  

Where:

* $x_{f}$: The original FP32 value (weight or activation).  
* $x_{q}$: The quantized integer value (e.g., INT8, typically in range $[-128, 127]$).  
* $S$: The scaling factor (a positive floating-point number).  
* $Z$: The zero point (an integer offset) that maps the floating-point value $0.0$ exactly to an integer in the quantized range.

The reverse process (de-quantization) is:

$$
x_{f} \approx (x_{q} - Z) \cdot S
$$

### **B. Quantization Range and Error**

Quantization introduces two main types of error:

1. **Clipping Error:** Occurs when the FP32 values **outlier** beyond the chosen quantization range ($[R_{min}, R_{max}]$). These outliers are clipped to the min/max integer values. This is why **Range Estimation** is critical.  
2. **Rounding Error:** Occurs when the scaled value $(\frac{x_{f}}{S} + Z)$ is rounded to the nearest integer.

### **C. Range Estimation Techniques (Calibration)**

Selecting the correct range $[R_{min}, R_{max}]$ for the floating-point values minimizes clipping error. This process is called **Calibration** when performed in Post-Training Quantization (PTQ).

* **Min-Max:** The simplest method. Sets $R_{min} = \min(x_{f})$ and $R_{max} = \max(x_{f})$. Highly sensitive to outliers.  
* **Percentile/Entropy/MSE:** More advanced methods that exclude extreme outliers (e.g., clipping the top 0.1% of values) or optimize the range to minimize the mean squared error (MSE) between the original and de-quantized distribution.

### **D. Symmetric vs. Asymmetric Quantization**

| Type | Range | Zero Point (Z) | Purpose |
| :---- | :---- | :---- | :---- |
| **Symmetric** | Centered around zero (e.g., $[-R, +R]$) | $Z = 0$ | Weights are often symmetrically distributed. Simpler to implement. |
| **Asymmetric** | Arbitrary range (e.g., $[R_{min}, R_{max}]$) | $Z \neq 0$ | Activations (which are often strictly positive, or *unsigned*) require $Z \neq 0$ to map $0.0$ correctly. More general. |

## **3\. Types of Quantization Deployment**

### **A. Post-Training Quantization (PTQ)**

* **Process:** Quantization is applied **after** the model has been fully trained in FP32.  
* **Pro:** Simple, fast, and does not require the original training data or infrastructure.  
* **Con:** Can suffer a slight drop in accuracy, especially at very low bit-widths (INT4 or lower), because the model never learned to compensate for the quantization error.  
* **Calibration:** Requires a small set of *unlabeled* data (calibration set) to estimate the min/max ranges for activations.

### **B. Quantization-Aware Training (QAT)**

- **Process:** The quantization operation (including rounding and clipping) is simulated during the original training process using a technique called **Fake Quantization**.  
- **Pro:** Achieves the highest accuracy at low bit-widths (e.g., INT4/INT2) because the network's optimizer adjusts the weights to compensate for the quantization error.  
- **Con:** More complex, takes longer, requires access to the training pipeline, and the original training data.

## References

- EfficientML.ai Course | 2023 Fall | MIT 6.5940: [ Complete course video series ](https://youtube.com/playlist?list=PL80kAHvQbh-pT4lCkDT53zT8DKmhE0idB&si=Uu00N0zKopEixhw3).

## Additional Resources
- [Quantization in Depth - A Crash Course by Hugging Face (DeepLearning.ai)](https://github.com/afondiel/Quantization-in-Depth-HF)
- [Quantization Fundamentals with Hugging Face (DeepLearning.ai Course)](https://github.com/afondiel/Quantization-Fundamentals-with-HF)
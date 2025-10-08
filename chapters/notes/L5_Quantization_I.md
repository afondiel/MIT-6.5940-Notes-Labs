# Lecture 5: Quantization (Part I)

- **Lecturers:** Professor Song Han
- **Date:** Fall 2023
- **Corresponding Course Website Section:** efficientml.ai

## 1. 🎯 Why It Matters for Edge AI

* **The Core Problem:** Deep learning models are typically trained using 32-bit floating point numbers (**FP32**). An FP32 number requires 4 bytes of memory, and its operations demand high compute complexity. Edge devices, especially microcontrollers (MCUs) and dedicated AI accelerators (NPUs), are optimized for **low-precision integer operations** (e.g., INT8).
* **Edge AI Benefits:** **Quantization** is the process of mapping these FP32 weights and activations to low-bit representations (e.g., **INT8**).
    * **4x Memory Reduction:** INT8 weights require 1/4 the memory of FP32.
    * **4x Bandwidth Reduction:** Faster data transfer (crucial for Memory-Bound models).
    * **High Compute Speedup:** Modern NPUs can perform integer MAC operations significantly faster and with much less power than FP32 operations.

---

## 2. 📝 Key Concepts and Theory

* **Definition & Overview:** Quantization is the method of mapping a continuous (or large finite) set of floating-point values to a smaller, discrete set of integer values.
* **Quantization Function (Symmetric Affine Quantization):**
    $$x_q = \text{round} \left( \frac{x}{S} + Z \right)$$
    Where:
    * **$x$:** The original FP32 value (weight or activation).
    * **$x_q$:** The quantized integer value (e.g., INT8, INT4).
    * **$S$ (Scale Factor):** A positive scaling value that determines the range represented by the integers.
    * **$Z$ (Zero Point):** An integer offset that maps the FP32 value $0$ to a specific integer.
* **Key Components of Quantization:**
    1.  **Bit-width:** The number of bits used (e.g., 8-bit, 4-bit, 2-bit). Lower is more efficient but risks higher accuracy loss.
    2.  **Range (Clipping):** The decision on which range of FP32 values $[\min, \max]$ will be mapped to the target integer range $[I_{\min}, I_{\max}]$. Values outside this range are "clipped."
    3.  **Mode (Symmetric vs. Asymmetric):**
        * **Symmetric:** Maps the FP32 range $[-A, +A]$ to the integer range (e.g., $[-127, 127]$ for signed INT8). $Z$ is 0.
        * **Asymmetric:** Maps the FP32 range $[A, B]$ to the integer range (e.g., $[0, 255]$ for unsigned INT8). $Z$ is non-zero, allowing $0$ to be represented exactly. **Asymmetric is often used for activations.**
* **Quantization Granularity:**
    * **Per-Tensor:** A single $S$ and $Z$ for all elements in a tensor (simple, common).
    * **Per-Axis/Per-Channel:** A separate $S$ and $Z$ for each output channel (more complex, but better accuracy, common for weights).

---

## 3. ⚙️ Practical Implementation & Tools

* **Implementation Steps (Post-Training Quantization - PTQ):** This is the simplest and most common form for edge deployment.
    1.  **Train in FP32:** Get the converged baseline model.
    2.  **Calibration:** Run the FP32 model on a small, representative subset of the training data (the **calibration set**).
    3.  **Collect Statistics:** For each layer's weights and activations, collect statistics (min/max or histograms) to determine the optimal $S$ and $Z$.
    4.  **Convert:** Use the computed $S$ and $Z$ to convert the FP32 weights/activations to INT8, generating the final quantized model.
* **Tools:**
    * **TensorFlow Lite Converter (TFLite):** The primary tool for PTQ, particularly for mobile and embedded devices.
    * **PyTorch Quantization:** Provides APIs for both PTQ and QAT (Quantization-Aware Training).

---

## 4. ⚖️ Trade-offs and Real-World Impact

* **Trade-off:** Quantization yields the best hardware speedup but is the technique most prone to **accuracy degradation**, particularly in activation layers where outliers can heavily skew the quantization range.
* **Hardware Dependence:** Quantization is only truly beneficial when the target device has a compatible **INT8 (or lower) instruction set**. Running a quantized model on a general-purpose CPU without INT8 support may require de-quantizing to FP32, negating all benefits.
* **PTQ vs. QAT:** PTQ is fast and requires no retraining, but its accuracy can suffer. This is the motivation for Lecture 6.

---

## 5. 🧪 Hands-on Lab Preview

* **What you will do:** Take a pre-trained FP32 MobileNetV2 model and apply **Post-Training Dynamic Range Quantization** (PTQ-DR) using TensorFlow Lite. You will profile the latency and memory reduction compared to the original FP32 model.
* **Key Skill Acquired:** Mastering the PTQ workflow, understanding the role of the calibration dataset, and using a model converter tool for deployment format generation.

# Lecture 6: Quantization (Part II)

- **Lecturers:** Professor Song Han
- **Date:** Fall 2023
- **Corresponding Course Website Section:** efficientml.ai

## 1. 🎯 Why It Matters for Edge AI

* **The Core Problem:** Post-Training Quantization (PTQ) can lead to unacceptable accuracy loss, especially for very deep or complex models (like LLMs or Vision Transformers) or when going to ultra-low bit-widths (e.g., INT4, ternary). The model was never "trained" to be robust to the large errors introduced by quantization.
* **Edge AI Benefits:** **Quantization-Aware Training (QAT)** simulates the quantization process *during* the training phase. This allows the model to learn weights that are inherently robust and "quantization-friendly," recovering almost all the accuracy lost in PTQ while maintaining the full efficiency benefits.

---

## 2. 📝 Key Concepts and Theory

* **Quantization-Aware Training (QAT):**
    * **Mechanism:** The forward pass of the training loop uses a **fake quantization** (simulating the rounding and clipping of the target integer format) to estimate the quantization errors. The backward pass uses the standard full-precision gradients (often using the **Straight-Through Estimator - STE**) to update the full-precision weights.
    * **Straight-Through Estimator (STE):** Since the `round` operation is non-differentiable, the STE is used in the backward pass. It acts as the identity function for the gradient: $\frac{\partial L}{\partial w} \approx \frac{\partial L}{\partial w_q}$.
* **The Forward Pass (Fake Quantization):**
    $$x_q = \text{clip} \left( \text{round} \left( \frac{x}{S} + Z \right), I_{\min}, I_{\max} \right)$$
    The model *computes* with $x_q$ but *stores* and *updates* $x$.
* **Other Advanced Quantization Techniques:**
    * **Mix-Precision Quantization:** Using different bit-widths (e.g., INT8 for some layers, FP16 for critical layers) based on the layer's sensitivity to quantization error.
    * **Quantization for LLMs:** Specialized techniques like **SmoothQuant** or **AWQ (Activation-aware Weight Quantization)** which address the challenge of highly skewed activation distributions in transformers.

---

## 3. ⚙️ Practical Implementation & Tools

* **Implementation Steps (QAT):**
    1.  **Insert Fake Quant Ops:** Modify the network by inserting "Fake Quantization" nodes after every layer that needs to be quantized.
    2.  **Fine-tune:** Retrain the model for a few epochs (usually fewer than the original training) using a low learning rate. The model learns to adjust its weights to minimize the error caused by the simulated rounding/clipping.
    3.  **Final Conversion:** Remove the Fake Quant ops and convert the fine-tuned full-precision weights to the final INT8 format.
* **Tools:**
    * **PyTorch QAT Module:** Offers the most flexible and robust implementation for QAT.
    * **TensorFlow Keras Quantization API:** Used to add quantization wrappers to layers for QAT.

---

## 4. ⚖️ Trade-offs and Real-World Impact

* **PTQ vs. QAT Trade-off:**
    * **PTQ:** Fast development time, no retraining cost, moderate accuracy.
    * **QAT:** Best accuracy, guaranteed deployment success, but requires a small amount of retraining time and access to the training dataset. **QAT is generally preferred for high-accuracy Edge AI systems.**
* **Bit-width Selection:** The choice between INT8, INT4, or even binary/ternary depends on the accuracy requirement. **INT8 is the industry standard** for general deployment, as it offers the best balance of speed and fidelity.
* **Deployment Barrier:** Even with QAT, some complex operations (e.g., custom layers or unusual activation functions) might not be supported by the target device's runtime/accelerator, which requires manual kernel optimization or model refactoring.

---

## 5. 🧪 Hands-on Lab Preview

* **What you will do:** Implement **Quantization-Aware Training** on a compressed MobileNetV2. You will compare the accuracy of the QAT model against the PTQ model (from Lab 5) and the original FP32 model, demonstrating the superior performance recovery of QAT.
* **Key Skill Acquired:** Utilizing the Straight-Through Estimator concept for effective training in the presence of non-differentiable operations and producing a near-lossless highly efficient model.

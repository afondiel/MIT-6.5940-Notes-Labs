# Lecture 15: Advanced Sparsity and Hardware Integration

## Quick Reference

|Item|Reference|
|---|---|
| Slides | [View Slides](https://drive.google.com/drive/folders/1A3P6IBuS8wCzLlpdRiQBO9b1uoK3pnPf?usp=sharing)|
| Video | [EfficientML.ai Lecture 15](http://www.youtube.com/watch?v=6cAmS-_vEh8)  |
|Lab| -- |
|Professor|[Song Han](https://github.com/songhan)|


## 1. 🎯 Why It Matters for Edge AI

* **The Core Problem:** Unstructured sparsity (Lecture 3) offers the highest compression but fails to deliver speedup on standard hardware due to irregular memory access. Structured sparsity (Lecture 4) guarantees speedup but sacrifices compression ratio. The goal is to find a middle ground: achieving **high sparsity** while ensuring **hardware-accelerated execution**.
* **Edge AI Benefits:** Advanced sparsity techniques enable the use of specialized **sparse AI accelerators** (e.g., dedicated NPUs or the latest GPUs/CPUs with sparsity support) to realize the full potential of pruning—getting both massive model compression *and* significant latency reduction.

---

## 2. 📝 Key Concepts and Theory

* **Coarse-Grained Sparsity (Block Sparsity):**
    * **Mechanism:** Instead of removing individual weights (unstructured) or full channels/filters (structured), weights are zeroed out in small, predefined **blocks** (e.g., $2 \times 4$ or $16 \times 1$).
    * **Hardware Advantage:** This creates a pattern that is regular enough for specialized hardware to efficiently skip zero blocks while still offering much higher compression than filter-level structured pruning.
    * **Example (NVIDIA Sparsity):** NVIDIA Ampere and newer architectures support **2:4 structured sparsity**: for every four consecutive weights in an output channel, at least two must be zeros. This provides a guaranteed 2x compression and 2x throughput boost with only minor accuracy loss.
* **The Importance of Weight Layout:** The physical arrangement of the weights in memory is crucial. Sparse operations often require **Compressed Sparse Row (CSR)** or **Compressed Sparse Column (CSC)** formats to minimize memory overhead from storing index pointers.
* **The "Lottery Ticket Hypothesis" Revisited (LTH):**
    * **Iterative Magnitude Pruning (IMP):** The most common way to find sparse subnetworks. It involves iteratively training, pruning, and fine-tuning.
    * **Rewinding/Retraining:** LTH suggests that the final performance of the sparse model relies heavily on the **initialization** of the *remaining* weights. Rewinding these weights to their values early in the dense training run can significantly improve sparse model performance.

---

## 3. ⚙️ Practical Implementation & Tools

* **Implementation Steps (2:4 Sparsity):**
    1.  **Dense Training:** Train the full model.
    2.  **Sparsity Enforcement:** Apply a simple filter to the weights, enforcing the 2:4 pattern (zeroing out the smallest two magnitude weights in every four-weight block).
    3.  **Fine-tuning:** Retrain the model to recover accuracy.
    4.  **Hardware Deployment:** The model is converted into a format that the target hardware's runtime understands as a sparse tensor, enabling the use of dedicated sparse matrix multiplication (SpGEMM) kernels.
* **Tools:**
    * **NVIDIA CUTLASS/cuSPARSE:** Libraries providing highly optimized kernels for sparse tensor operations, crucial for utilizing sparsity on NVIDIA hardware.
    * **Pruning APIs (PyTorch/TF):** Provide utilities to enforce various structural and coarse-grained sparsity patterns.

---

## 4. ⚖️ Trade-offs and Real-World Impact

* **Trade-off: Sparsity vs. Hardware Specificity:** The best speedups come from sparsity patterns that **perfectly match** the target hardware (e.g., 2:4 for NVIDIA, or specific block sizes for other NPUs). This creates vendor lock-in but yields maximum efficiency.
* **The Sweet Spot:** Coarse-grained sparsity strikes the optimal balance for modern platforms: high compression approaching unstructured pruning's level, but with the **guaranteed speedup** of a structured pattern.
* **Energy Efficiency:** Skipping computations for zero weights not only reduces latency but drastically reduces the power consumption of the device, which is a major advantage for battery-powered Edge AI.

---

## 5. 🧪 Hands-on Lab Preview

* **What you will do:** Implement **Block Sparsity** (e.g., $2 \times 2$ block zeroing) on a CNN layer. You will compare the compression ratio and the regularity of the resulting weight tensor against both unstructured and full filter pruning.
* **Key Skill Acquired:** Understanding the computational and memory advantages of **coarse-grained, hardware-aware sparsity** and the process of transforming a dense weight matrix into a deployable sparse format.


## References

- EfficientML.ai Course | 2023 Fall | MIT 6.5940: [ Complete course video series ](https://youtube.com/playlist?list=PL80kAHvQbh-pT4lCkDT53zT8DKmhE0idB&si=Uu00N0zKopEixhw3).
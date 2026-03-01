# Lecture 18: Distributed Training (Part II) - Advanced Techniques

## Quick Reference

|Item|Reference|
|---|---|
| Slides | [Slides](https://drive.google.com/drive/folders/1A3P6IBuS8wCzLlpdRiQBO9b1uoK3pnPf?usp=sharing)|
| Video | [EfficientML.ai Lecture 18](http://www.youtube.com/watch?v=6cAmS-_vEh8)  |
|Lab| -- |
|Professor|[Song Han](https://github.com/songhan)|


## 1. 🎯 Scaling Up: Hybrid Parallelism for LLMs

For models like GPT-3/4 or LLaMA, which contain hundreds of billions of parameters, a single strategy is not enough. Training requires combining multiple forms of parallelism, known as **Hybrid Parallelism**.

### 1.1. Tensor Parallelism (TP)
* **Concept:** Splits the computation **within a single layer** (intra-layer parallelism). For example, a large Matrix Multiplication ($W \cdot X$) is split into two smaller matrix multiplications ($W_1 \cdot X_1$ and $W_2 \cdot X_2$) across two GPUs.
* **Mechanism:** Weights ($W$) and activations ($X$) are sharded (split) across devices. After multiplication, the partial results are synchronized using an **All-Reduce** operation to reconstruct the final output before the next non-linear operation.
* **Use Case:** Essential when a **single layer** is too large to fit on one GPU (e.g., the large Feed-Forward or Attention blocks in Transformers). Requires **high-bandwidth** communication (e.g., NVLink) because it involves communication on *every layer*.

### 1.2. Pipeline Parallelism (PP)
* **Concept:** Splits the model **across layers** (inter-layer parallelism), like the assembly line analogy.
* **Mechanism:** Layer 1 is on GPU 1, Layer 2 is on GPU 2, and so on. To keep the GPUs busy, the data batch is split into **micro-batches ($\mu B$)**. GPU 1 computes its forward pass for $\mu B_1$, sends the activation to GPU 2, then immediately starts working on $\mu B_2$.
* **Use Case:** Ideal for splitting **very deep models** across nodes/servers with **slower inter-node network bandwidth**.
* **The Catch:** Requires complex scheduling (e.g., **GPipe, PipeDream**) to minimize the **"Pipeline Bubble"**—the idle time at the beginning and end of each batch when GPUs must wait for previous stages to finish.

---

## 2. 📝 Zero Redundancy Optimizer (ZeRO / FSDP)

Standard Data Parallelism (DDP) is inefficient for large models because it replicates the entire model *and* the optimizer state (which can be $12 \times$ the model size for Adam) on *every* GPU.

**ZeRO (Zero Redundancy Optimizer)** and its PyTorch implementation, **FSDP (Fully Sharded Data Parallel)**, eliminate this memory redundancy by sharding (partitioning) the memory components of the training process across the data-parallel GPUs.

| Stage | Component Sharded | Memory Saved (Factor) | Purpose |
| :---: | :---: | :---: | :---: |
| **ZeRO-1** | **Optimizer State** ($P_{\text{opt}}$) | $\sim 4 \times$ | Distributes the massive memory footprint of optimizers (like Adam/AdamW). |
| **ZeRO-2** | **$P_{\text{opt}}$ + Gradients** ($G$) | $\sim 8 \times$ | Further reduces memory by partitioning the gradients. |
| **ZeRO-3 / FSDP** | **$P_{\text{opt}}$ + $G$ + Parameters** ($W$) | $\sim \mathbf{3} \mathbf{N} \mathbf{\times}$ | The **most effective** stage. It shards the entire model weights, allowing models larger than a single GPU's memory to be trained using Data Parallelism. |

* **How FSDP Works (Key Insight):**
    * During the forward pass of a specific layer, FSDP uses an **All-Gather** operation to temporarily reconstruct the full parameters ($W$) for that layer on the device.
    * After the layer computation is done, the full parameters are **immediately freed** (released from memory).
    * This ensures that at any given time, a GPU only holds the full parameters of the layer currently being computed.

---

## 3. ⚙️ Communication Efficiency Techniques

Since communication is the main bottleneck, several techniques are used to reduce the size or frequency of data transfer.

### 3.1. Mixed-Precision Training
* **Mechanism:** Use **FP16 (Half-Precision)** or **bfloat16** for weights and activations, while keeping a full-precision (FP32) copy for the master weights (to prevent precision loss during updates).
* **Benefit:** Reduces memory usage by $\mathbf{2 \times}$ and communication volume by $\mathbf{2 \times}$. It's a standard, highly effective technique.

### 3.2. Gradient Compression
* **Concept:** Reduce the size of the gradients transferred during the All-Reduce step.
* **Sparsification (Top-K):** Only transmit the **Top-K** percent of gradient elements with the largest magnitude, setting the rest to zero.
    * *Trade-off:* Saves massive bandwidth but the discarded information must be handled using **Error Feedback** (or residual accumulation) where the discarded gradients are stored locally and added to the next iteration's gradient.
* **Quantization (TernGrad/QSGD):** Quantize the gradient values to a few bits (e.g., 1-bit or 2-bit) before transmission.
    * *Trade-off:* High compression ratio, but requires careful mathematical proof (and implementation) to ensure convergence stability.

### 3.3. Overlapping Communication and Computation
* **Concept:** Don't wait for communication to finish before starting the next computation, and vice versa.
* **Mechanism:** In DDP (Distributed Data Parallel), the framework can compute the gradients for later layers while simultaneously performing the **All-Reduce** operation on the gradients of earlier layers that have already been computed. This uses the **network bandwidth** and the **GPU compute cores** at the same time, maximizing utilization.

---

## 4. ⚖️ Deployment Frameworks

Modern deep learning frameworks handle these complexities transparently:

* **PyTorch Distributed Data Parallel (DDP):** The standard, robust implementation of Synchronous Data Parallelism.
* **NVIDIA Apex/Megatron-LM:** Tools that introduced many of the first practical implementations of **Tensor Parallelism** and advanced Mixed-Precision for Transformer models.
* **PyTorch FSDP:** The state-of-the-art framework for sharding parameters, gradients, and optimizer states, making **Data Parallelism** memory-efficient enough for models up to hundreds of billions of parameters.
* **Microsoft DeepSpeed:** A comprehensive optimization library that includes the **ZeRO** stages, as well as complex Pipeline Parallelism schedulers.


## References

- EfficientML.ai Course | 2023 Fall | MIT 6.5940: [ Complete course video series ](https://youtube.com/playlist?list=PL80kAHvQbh-pT4lCkDT53zT8DKmhE0idB&si=Uu00N0zKopEixhw3).
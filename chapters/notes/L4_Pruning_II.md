# Lecture 4: Pruning and Sparsity (Part II)

- **Lecturers:** Professor Song Han
- **Date:** Fall 2023
- **Corresponding Course Website Section:** efficientml.ai

## 1. 🎯 Why It Matters for Edge AI

* **The Core Problem:** Unstructured pruning yields high memory compression but low real-world speedup because the irregular zero-patterns require complex indexing, often resulting in inefficient memory operations.
* **Edge AI Benefits:** **Structured Pruning** removes entire blocks of computation (e.g., filters, channels, heads). This creates a **smaller, dense model** with regular structure, which can be directly run on highly optimized, standard dense deep learning libraries (cuDNN, TFLite) for guaranteed **latency reduction** on almost all edge hardware (CPUs/GPUs/NPUs).

---

## 2. 📝 Key Concepts and Theory

* **Structured Pruning (Hardware-Friendly Pruning):**
    * **Mechanism:** Removes entire groups of parameters (e.g., removing a full convolutional filter or a channel/neuron).
    * **Types of Structured Pruning:**
        1.  **Filter Pruning:** Removing entire 3D filters from a convolutional layer.
        2.  **Channel Pruning:** Removing the associated input/output channels across multiple layers.
        3.  **Block Pruning:** Removing entire matrix blocks in FC layers.
* **Criterion (Filter Importance):** Instead of individual weights, the importance of an entire filter (or channel) is measured. A common method is to use the **L2-norm** or **L1-norm** of the entire filter's weight vector.
    $$\text{Importance}(f_i) = ||\mathbf{w}_i||_2 \quad \text{or} \quad ||\mathbf{w}_i||_1$$
* **Pruning Granularity:** The level at which sparsity is enforced (e.g., 4:2 structured pruning means for every 4 values, at least 2 must be kept, which can be efficiently processed by specialized hardware like NVIDIA's Tensor Cores).

---

## 3. ⚙️ Practical Implementation & Tools

* **Implementation Steps (Filter Pruning):**
    1.  **Calculate Importance:** Compute the L1-norm of every filter's weight tensor.
    2.  **Sort and Select:** Sort the filters by their norm and select the filters with the lowest norms to be removed.
    3.  **Rebuild Model:** Instead of masking zeros, literally *remove* the selected filters and their corresponding channels in the next layer, creating a physically smaller, **dense** network.
    4.  **Fine-tune:** Train the new, smaller network.
* **The Problem of Layer-Wise vs. Global Pruning:**
    * **Layer-Wise:** Pruning a fixed percentage from each layer; simple but suboptimal.
    * **Global:** Pruning across the entire network based on a single threshold; allows for better performance by removing more from the most redundant layers and less from the most critical layers. **Global Pruning** is preferred.

---

## 4. ⚖️ Trade-offs and Real-World Impact

* **Trade-off:** Structured pruning achieves **lower overall compression** (as it can't reach the high sparsity of unstructured pruning) but guarantees a **real, measurable speedup** on existing edge hardware, making it far more practical for Edge AI deployment.
* **Practical Frameworks:** Tools like **Nvidia's APEX** or hardware-aware frameworks often enforce structured or block-level sparsity to maximize the utilization of underlying AI accelerators.
* **The Importance of Fine-Tuning:** Structured pruning can cause a larger initial accuracy drop than unstructured pruning because it removes entire feature extractors. A robust fine-tuning schedule is critical for recovering performance.

---

## 5. 🧪 Hands-on Lab Preview

* **What you will do:** Implement a **Channel Pruning** technique. You will calculate the L1-norm of filters, prune the lowest-scoring channels, and verify that the resulting model has a reduced number of MACs and a faster real-world inference latency.
* **Key Skill Acquired:** Creating a **hardware-friendly** model by restructuring the network architecture post-pruning and quantifying the guaranteed **latency reduction** on a simulated edge environment.

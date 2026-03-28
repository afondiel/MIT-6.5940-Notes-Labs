# **Lesson 3: Pruning and Sparsity (Part I)** 

focuses on the foundational concepts, formulations, and methodologies for reducing the size and computational cost of neural networks through pruning.

### **I. Motivation and Definition**
*   **The Problem:** Modern deep learning models are outgrowing hardware capabilities; for example, model sizes are growing 4x every two years while hardware capacity only doubles.
*   **Energy Efficiency:** Data movement is the most expensive operation in terms of energy; accessing **DRAM** consumes roughly **200 times** more energy than a 32-bit integer multiplication.
*   **Concept:** Pruning involves making a neural network smaller by **removing redundant synapses and neurons**, a process that occurs naturally in the human brain during development from infancy to adulthood.

### **II. Pruning Formulation**
*   **Mathematical Objective:** Pruning is generally formulated as an optimization problem: $\text{arg min}_{W_P} L(x; W_P)$ subject to $||W_P||_0 \le N$, where $L$ is the loss function, $W_P$ represents the pruned weights, and $N$ is the target number of non-zero elements.

### **III. Pruning Granularity**
Pruning can be performed in various patterns, ranging from irregular to highly structured:
*   **Fine-grained (Unstructured):** Offers the most flexibility and the highest compression ratios but is **difficult to accelerate** on standard hardware like GPUs due to its irregular memory access.
*   **Pattern-based (N:M Sparsity):** A middle ground where, for every $M$ contiguous elements, $N$ are pruned. A notable example is **2:4 sparsity**, which is supported by NVIDIA Ampere architecture for up to **2x speedup**.
*   **Structured (Channel/Neuron):** Involves removing entire channels or neurons. While this offers **direct hardware speedup** and easier deployment, it typically results in a lower overall compression ratio compared to fine-grained pruning.

### **IV. Pruning Criteria**
How to determine which weights are "least important" to remove:
*   **Magnitude-based:** A heuristic approach that assumes weights with **smaller absolute values** are less important ($Importance = |W|$).
*   **Second-Order-based (Optimal Brain Damage):** Uses the **Taylor series expansion** of the loss function to minimize the increase in error. It assumes that synapses with a smaller product of the weight squared and the **Hessian matrix** value ($h_{ii}w_i^2$) should be removed.

### **V. The Pruning Pipeline**
*   **Iteration:** Pruning is most effective when done **iteratively**. The process typically involves training the original network, pruning a portion of the weights, and then **fine-tuning** the remaining weights to recover accuracy.
*   **Sensitivity Analysis:** To find per-layer pruning ratios, engineers often perform sensitivity analysis by pruning each layer individually at various ratios and observing the accuracy degradation. This helps identify which layers are more "redundant" and which are more "sensitive."

## References
- Source: https://notebooklm.google.com/notebook/39b7bd5c-37ef-451c-83dc-98b62ec8dc04
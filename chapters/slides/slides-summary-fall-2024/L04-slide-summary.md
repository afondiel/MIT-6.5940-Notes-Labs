# **Lesson 4: Pruning and Sparsity (Part II)** 

focuses on advanced techniques for determining pruning ratios, automated compression workflows, and the system/hardware support required to accelerate sparse neural networks.

### **I. Pruning Formulation and Ratios**
Pruning is mathematically formulated as an optimization problem: $\text{arg min}_{W_P} L(x; W_P)$ subject to $||W_P||_0 \le N$, where the goal is to minimize loss while maintaining a target number of non-zero parameters. Because different layers in a network have varying levels of redundancy, finding the right **pruning ratio** for each layer is critical.
*   **Sensitivity Analysis:** This manual process involves pruning one layer at a time to observe how accuracy degrades as the pruning ratio increases. It helps identify "sensitive" layers (where small pruning causes high accuracy loss) and "redundant" layers.
*   **Limitations:** Sensitivity analysis often ignores the **interactions between layers**, which can lead to sub-optimal compression when applied globally.

### **II. Automatic Pruning (AutoML)**
To overcome manual tuning, automated solutions use algorithms to find the best per-layer ratios:
*   **AMC (AutoML for Model Compression):** This method uses **Reinforcement Learning** (specifically a DDPG agent) to propose pruning ratios. The agent receives a reward based on accuracy and resource constraints (like FLOPs or latency) to learn the optimal policy. 
*   **NetAdapt:** A rule-based iterative approach that targets specific hardware constraints, such as latency or energy. In each iteration, it prunes the layer that yields the desired resource reduction while maintaining the highest possible accuracy.

### **III. Training and Fine-Tuning**
After pruning, accuracy often drops, especially at high sparsity ratios.
*   **Fine-tuning:** Accuracy is recovered by retraining the remaining weights, typically using a lower learning rate (1/10th or 1/100th of the original).
*   **Iterative Pruning:** Rather than pruning to the target sparsity in one step, **iterative pruning and fine-tuning** cycles gradually increase sparsity, which is significantly more effective for preserving model performance.

### **IV. System and Hardware Support**
Pruning creates **sparsity**, which requires specialized hardware and software to achieve real-world speedups.
*   **EIE (Efficient Inference Engine):** The first DNN accelerator to exploit both **weight and activation sparsity**, skipping zero-value computations and using weight sharing to reduce memory footprint.
*   **M:N Sparsity (2:4):** Supported by **NVIDIA's Ampere GPU architecture**, this "structured" fine-grained sparsity requires that for every four contiguous weights, at least two must be zero. This enables up to **2x peak performance speedup** while maintaining accuracy.
*   **TorchSparse:** A library optimized for **Sparse Convolution**, which is essential for processing high-sparsity 3D data like LiDAR point clouds used in autonomous driving. It avoids redundant computation in physical space where many voxels are empty.

## References
- Source: https://notebooklm.google.com/notebook/39b7bd5c-37ef-451c-83dc-98b62ec8dc04
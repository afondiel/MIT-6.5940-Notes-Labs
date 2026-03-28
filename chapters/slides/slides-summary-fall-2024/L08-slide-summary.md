# **Lesson 8: Neural Architecture Search (Part II)** 

focuses on moving from expensive, manual, or proxy-based architecture design to efficient, hardware-aware automated search that can be deployed across diverse platforms.

### **I. Beyond Proxy Tasks: ProxylessNAS**
Traditional NAS methods were computationally prohibitive (e.g., NASNet required 48,000 GPU hours) and relied on **proxy tasks**, such as searching on smaller datasets like CIFAR-10 and then transferring the architecture to ImageNet. **ProxylessNAS** overcomes this by allowing direct search on the **target task and hardware**, utilizing profiled latency instead of proxy metrics like FLOPs to find truly efficient models.

### **II. Once-for-All (OFA) Network**
The OFA framework introduces the "train once, get many" philosophy to reduce the massive carbon footprint and computational cost of training separate models for every device.
*   **Concept:** A single large **super-network** is trained to contain many child networks ($10^{19}$ candidates) that are sparsely activated and **share weights** with the main network.
*   **Specialization:** This single OFA network can then be used to extract specialized models for diverse hardware platforms (e.g., different generations of Samsung phones, microcontrollers, or even varying battery levels) without any additional retraining.
*   **Progressive Shrinking (PS):** This is the core training algorithm for OFA. It starts by training the full network and then **progressively prunes** dimensions to introduce elasticity in:
    *   **Kernel Size:** Using a transformation matrix to take centered weights for smaller kernels.
    *   **Depth:** Gradually allowing later layers in a unit to be skipped.
    *   **Width:** Sorting channels by importance and progressively shrinking the number of active channels.
    *   **Resolution:** Randomly sampling input image sizes for each batch during training.

### **III. Zero-Shot NAS**
To further accelerate the search process, **Zero-Shot NAS** techniques estimate model accuracy **without any training**.
*   **ZenNAS:** Analyzes the architecture by measuring its sensitivity to input perturbations. A "Zen score" is calculated based on log differences and batch normalization variance, assuming better models are more sensitive to input changes.
*   **GradSign:** Uses gradient sign information at initialization to infer the performance of candidate architectures.

### **IV. Specialized Applications of NAS**
The lecture highlights the versatility of these techniques across different domains:
*   **Quantum AI (QuantumNAS):** Searches for robust circuit architectures that can withstand **quantum noise**, which typically causes a massive drop in accuracy (e.g., from 87% to 47%).
*   **Large Language Models (Flextron):** Implements a "many-in-one" flexible LLM architecture that adapts to user-defined latency and accuracy targets during inference with the same set of weights.
*   **Visual Tasks:** Includes **LitePose** for efficient on-device pose estimation, **Anycost GANs** for interactive image synthesis, and specialized 3D architectures for LiDAR point clouds.

## References
- Source: https://notebooklm.google.com/notebook/39b7bd5c-37ef-451c-83dc-98b62ec8dc04
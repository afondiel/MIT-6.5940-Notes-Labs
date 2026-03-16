# **Lesson 20: Distributed Training (Part II)** 

expands on advanced strategies for scaling deep learning across multiple nodes, focusing on hybrid parallelism, identifying hardware bottlenecks, and techniques to overcome them through gradient compression and delayed updates.

### **I. Hybrid Parallelism and Auto-Parallelization**
Large models often require combining multiple parallelization strategies to balance memory and utilization.
*   **2D Parallelism:** Common combinations include **Data Parallelism + Pipeline Parallelism** (e.g., sharding data across groups that each host a layer-partitioned model) or **Pipeline Parallelism + Tensor Parallelism** (e.g., Megatron-LM).
*   **2D Sequence Parallelism:** Utilizes **all-to-all repartitioning** within nodes and **ring attention** between nodes to handle long-context training.
*   **Alpa Compiler:** An automated system that searches for optimal parallel strategies. It uses a hierarchical search space, applying **dynamic programming** for inter-operator parallelism (pipeline) and **integer linear programming** for intra-operator parallelism (tensor/data). Alpa can match or outperform specialized manual systems by up to 8x.

### **II. Bandwidth and Latency Bottlenecks**
Distributed training performance is heavily dictated by the communication network.
*   **Bandwidth Bottleneck:** Larger models and more training nodes increase the volume of data transferred during All-Reduce, which can be mitigated by hardware upgrades or compression.
*   **Latency Bottleneck:** High network latency (e.g., from wireless connections or long-distance data centers) forces GPUs to sit idle while waiting for synchronization. While bandwidth is relatively easy to improve, latency is limited by physical distance and networking overhead.

### **III. Gradient Compression**
To overcome the bandwidth bottleneck, the amount of data transmitted between nodes is reduced:
*   **Gradient Pruning (Sparse Communication):** Only the "top-k" gradients by magnitude are sent, while the rest are kept as a local residual. 
*   **Deep Gradient Compression (DGC):** Enhances sparse communication to maintain accuracy on modern models like ResNet using **momentum correction** (accumulating velocity instead of raw gradients), **gradient clipping**, and **warm-up training**. DGC achieves up to 600x compression (99.9% sparsity) with negligible accuracy loss.
*   **PowerSGD:** A low-rank gradient compression method that uses **matrix factorization** instead of fine-grained pruning to prevent gradients from becoming "denser" during the All-Reduce process.
*   **Gradient Quantization:** Reduces bit-width using methods like **1-Bit SGD** or **TernGrad**, which utilize scaling factors and error accumulation to preserve precision.

### **IV. Delayed Gradient Updates**
To address high-latency environments, **Delayed Gradient Averaging (DGA)** allows workers to overlap communication with computation.
*   **Mechanism:** Workers continue performing local updates while the averaged parameters from a previous iteration ($i-D$) are still in transmission. 
*   **Staleness Correction:** Because "stale" gradients can hurt performance, DGA applies correction terms to ensure the model remains accurate. This approach is particularly effective for **Federated Learning** and long-distance distributed training, where it can provide significant speedups.

## References
- Source: https://notebooklm.google.com/notebook/39b7bd5c-37ef-451c-83dc-98b62ec8dc04
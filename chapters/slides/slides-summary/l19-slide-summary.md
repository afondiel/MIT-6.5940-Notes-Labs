# **Lesson 19: Distributed Training (Part I)** 

provides a comprehensive overview of the techniques used to accelerate deep learning by scaling training across multiple hardware accelerators (GPUs).

### **I. Motivation for Distributed Training**
The primary goal of distributed training is to reduce the development and research cycle by parallelizing massive computational tasks. 
*   **Speedup:** Ideally, a task that takes **10 GPU days** could be finished in **14 minutes** using 1024 GPUs.
*   **Real-world Example (SUMMIT):** Using the SUMMIT supercomputer, researchers reduced the training time for a video model (TSM) on 660 hours of video from **49 hours (6 GPUs) to just 14 minutes (1536 GPUs)**, achieving a **211x speedup** with no loss in accuracy.

### **II. Data Parallelism**
Data parallelism is the most common distributed training method, where the dataset is split across multiple worker nodes while each node maintains a local replica of the model.
*   **The Parameter Server (PS):** A central controller manages the training process. 
*   **Workflow:**
    1.  **Replicate:** The model is pulled from the PS to each worker.
    2.  **Split:** The training dataset is randomly and evenly split among workers.
    3.  **Compute:** Each worker calculates **local gradients** based on its assigned data.
    4.  **Aggregate:** Workers "push" gradients to the PS, which sums them up.
    5.  **Update:** The PS performs the gradient step to update the master model weights, and the cycle repeats.
*   **Communication Primitives:** This process relies on **Broadcast** (PS to workers) and **Reduce** (workers to PS) operations, where the PS communication complexity is **$O(N)$** for $N$ workers.

### **III. Handling Large Models (Model Parallelism)**
When models are too large to fit into a single GPU's memory (e.g., GPT-3 175B requires ~350GB of memory, exceeding the 80GB capacity of an NVIDIA A100), the model itself must be partitioned.
*   **Pipeline Parallelism:** The model is split layer-wise across multiple GPUs. To prevent GPUs from sitting idle (the "bubble" problem), frameworks like **GPipe** divide input batches into smaller **micro-batches** to keep all devices busy.
*   **Tensor Parallelism:** This involves splitting the activations and weights within a single layer across GPUs. For example, in self-attention layers, different attention heads can be computed locally on different GPUs before being synchronized.
*   **Sequence Parallelism:** Specifically for long-context training, this method splits the input sequence of tokens across GPUs.

### **IV. Advanced Memory Optimization**
To further reduce the memory footprint of data parallelism, techniques like **ZeRO (1/2/3)** and **FSDP (Fully Sharded Data Parallelism)** are used. These methods shard the optimizer states, gradients, and parameters across all available GPUs instead of replicating them, allowing much larger models to be trained on existing hardware.

## References
- Source: https://notebooklm.google.com/notebook/39b7bd5c-37ef-451c-83dc-98b62ec8dc04
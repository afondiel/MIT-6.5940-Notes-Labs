# Lecture 17: Distributed Training (Part I) - Core Concepts

## Quick Reference

|Item|Reference|
|---|---|
| Slides | [View Slides](https://drive.google.com/drive/folders/1A3P6IBuS8wCzLlpdRiQBO9b1uoK3pnPf?usp=sharing)|
| Video | [EfficientML.ai Lecture 17](http://www.youtube.com/watch?v=6cAmS-_vEh8)  |
|Lab| -- |
|Professor|[Song Han](https://github.com/songhan)|


## 1. 🎯 Why It Matters for Efficiency

* **The Necessity:** Models are getting too big ($\text{LLaMA-3}$ has $\sim100$ billion parameters; training datasets are terabytes). Single GPUs cannot hold the model weights, the activations, and the optimizer states simultaneously.
* **The Solution:** **Distributed Training** scales the training process across multiple devices (GPUs/TPUs) and multiple machines (nodes) to reduce the **Time-to-Train (TTT)**, which is a major factor in efficiency and development cost.

---

## 2. 📝 Key Paradigms of Parallelism

Distributed training primarily relies on two fundamental techniques: **Data Parallelism** and **Model Parallelism**.

### 2.1. Data Parallelism (DP)
* **Concept:** The entire model is **replicated** across all available devices (workers). The training dataset is **split** into chunks, and each worker processes a different chunk simultaneously.
* **The Workflow:**
    1.  **Split Data:** The mini-batch is divided across $N$ devices.
    2.  **Local Compute:** Each device performs the Forward Pass and the Backward Pass to compute its local gradients.
    3.  **Synchronization (Crucial Step):** All local gradients are averaged to obtain the global gradient.
    4.  **Update Weights:** The global gradient is used to update the model copy on every device.
* **Use Case:** When the **model fits** on a single device, but the **dataset is too large** or the training time is too long. DP is the most common form of distributed training.

### 2.2. Model Parallelism (MP)
* **Concept:** The model itself is **split** across multiple devices. No device holds the complete model.
* **The Workflow:**
    1.  **Split Model:** Different layers or parts of layers are assigned to different devices.
    2.  **Sequential Compute:** Data (activations) flows sequentially through the devices, like an **assembly line**.
    3.  **Communication:** Communication happens when intermediate activations must be transferred between devices (e.g., from Layer 1 on GPU 1 to Layer 2 on GPU 2).
* **Use Case:** When the **model is too large** to fit into the memory of a single device (e.g., LLMs with hundreds of billions of parameters). This is often necessary but introduces communication and idle time bottlenecks.

---

## 3. ⚙️ Synchronization Schemes

The efficiency of Data Parallelism depends entirely on how the gradients are synchronized.

### 3.1. Synchronous SGD (Sync-SGD)
* **Mechanism:** All workers start the computation at the same time and **must wait** until the **slowest worker** completes its gradient calculation before the weights are updated.
* **Key Operation: All-Reduce:** A highly optimized collective communication primitive (e.g., implemented by NVIDIA's NCCL or Google's GLOO). It efficiently computes the sum (or average) of all local tensors across all workers and broadcasts the result back to all workers, resulting in identical weights on all devices.
* **Advantages:** **Guaranteed convergence** (equivalent to training with a global batch size of $N \times B$).
* **Disadvantages:** **Straggler problem**—the whole system is slowed down by the single slowest device/connection.

### 3.2. Asynchronous SGD (Async-SGD)
* **Mechanism:** Workers operate independently. A worker completes its gradient calculation and immediately sends its update to a central **Parameter Server (PS)**, which updates the master model without waiting for other workers.
* **Advantages:** **High utilization** (no waiting for stragglers) and potentially **faster wall-clock time**.
* **Disadvantages:** **Stale Gradients**—a fast worker might update weights based on model parameters that are already old because other workers have updated them since the fast worker started its forward pass. This can hurt convergence and result in a lower final accuracy.

---

## 4. ⚖️ Challenges and Bottlenecks

1.  **Communication Bottleneck:** The primary efficiency killer. Gradients (and activations in MP) must be exchanged over the network. The larger the model/batch size, the greater the volume of data to transmit.
    * *Solution:* High-speed interconnects (InfiniBand, NVLink) and techniques like **Gradient Compression** (covered in Part II).
2.  **Large Batch Optimization:** Sync-SGD effectively uses a huge global batch size, which can lead to poorer generalization. Training often requires specialized optimizers (e.g., **LARS, LAMB**) and careful learning rate scheduling to ensure convergence.
3.  **The "Straggler" Problem (Sync-SGD):** Heterogeneous hardware or network fluctuations can cause some workers to lag, idling the entire cluster.



## References

- EfficientML.ai Course | 2023 Fall | MIT 6.5940: [ Complete course video series ](https://youtube.com/playlist?list=PL80kAHvQbh-pT4lCkDT53zT8DKmhE0idB&si=Uu00N0zKopEixhw3).
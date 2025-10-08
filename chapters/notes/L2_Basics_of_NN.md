# Lecture 2: Basics of Neural Networks for Efficiency

- **Lecturers:** Professor Song Han
- **Date:** Fall 2023
- **Corresponding Course Website Section:** efficientml.ai

## 1\. 🎯 Why It Matters for Edge AI

* **The Core Problem:** To optimize an existing neural network, an engineer must first understand **where the bottleneck is**. Is the network compute-bound (too many FLOPs) or memory-bound (too many parameter transfers)? This lecture provides the foundational vocabulary.
* **Edge AI Benefits:** In Edge AI, every operation counts. Knowing the structure helps an engineer surgically apply optimization techniques. For instance, convolution layers are often **Compute-Bound**, while fully connected layers can be **Memory-Bound** due to heavy weight loading.

-----

## 2\. 📝 Key Concepts and Theory

* **Neural Network as a Computational Graph:** A DNN is a series of interconnected operations (nodes) and tensors (edges). Efficiency is about optimizing the nodes and the flow between them.
* **Core Building Blocks:**
    * **Convolutional Layer (Conv):** The workhorse of CNNs. Computational cost is dominated by the number of Multiply-Accumulate Operations (**MACs**). $MACs \propto \text{Output\_channels} \times \text{Kernel\_size}^2 \times \text{Input\_size}^2$.
    * **Fully Connected Layer (FC/Dense):** Each input connects to every output. FLOPs are $O(\text{Input\_size} \times \text{Output\_size})$.
* **Latency Breakdown (The $T_{total}$ Equation):** The total inference time is roughly the sum of computation time and memory access time.
  $$T_{\text{total}} \approx T_{\text{compute}} + T_{\text{memory}}$$
    * $T_{\text{compute}} \propto \frac{\text{Number of Operations (FLOPs)}}{\text{Hardware Peak Compute (OPS/s)}}$
    * $T_{\text{memory}} \propto \frac{\text{Data Movement (Bytes)}}{\text{Memory Bandwidth (Bytes/s)}}$
* **Key Insight: Latency vs. Throughput:**
    * **Throughput** can be increased by **batching** (coarse-grained parallelism, e.g., using more GPU cores).
    * **Latency** is harder to reduce, often requiring **fine-grained optimization** like overlapping compute with data movement (**memory-compute overlap**).

-----

## 3\. ⚙️ Practical Implementation & Tools

* **Implementation Steps (Performance Analysis):** Before optimizing, profile a baseline model. Identify the layers with the highest $T_{\text{total}}$ and determine if they are **Compute-Bound** (high FLOPs/MACs) or **Memory-Bound** (high data movement).
* **Industry Tools for Analysis:** Tools like **TensorFlow Profiler**, **PyTorch Profiler**, or vendor-specific profilers (e.g., Qualcomm Neural Processing SDK's profiling tools) are essential for this breakdown.

-----

## 4\. ⚖️ Trade-offs and Real-World Impact

* **The Right Metric:** A model with lower FLOPs might not be faster in the real world if it is heavily **Memory-Bound**. An efficient model must optimize the overall $T_{\text{total}}$.
* **Convolutional Optimization:** Techniques like using $1 \times 1$ convolutions (e.g., in MobileNets) are essential for reducing the FLOP-heavy $3 \times 3$ operations while managing the increase in channels.

-----

## 5\. 🧪 Hands-on Lab Preview

* **What you will do:** Implement basic building blocks of a neural network (e.g., a simple CNN) and use a profiler to measure the FLOPs/MACs and the real-world latency contribution of each layer.
* **Key Skill Acquired:** Quantifying the computational cost of a layer using the MACs/FLOPs formula and relating it to observed **latency** to diagnose the performance bottleneck.

The course notes will continue in this structured format for the remaining 21 lectures.

-----

You can learn more about the course on the official YouTube playlist: [EfficientML.ai Course Playlist (Fall 2023)](https://www.youtube.com/playlist?list=PL80kAHvQbh-pT4lCkDT53zT8DKmhE0idB). This link is the source of the entire lecture series you are using for your notes.
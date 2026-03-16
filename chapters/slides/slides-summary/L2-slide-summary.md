# **Lesson 2: Basics of Neural Networks** 


provides a foundational review of neural network terminology, common building blocks (layers), and essential efficiency metrics used to evaluate model performance on hardware.

### **I. Neural Network Terminology and Building Blocks**
*   **Fundamental Components:** The lecture identifies **neurons** as the source of **activations** (or features) and **synapses** as the representation of **weights** (or parameters). The width of a model is defined by the dimensionality of its hidden layers.
*   **Common Layers:**
    *   **Fully-Connected (Linear):** Each output neuron connects to every input neuron. Weights are typically stored as a $(c_o, c_i)$ tensor.
    *   **Convolutional:** Output neurons connect only to input neurons within a local **receptive field**. These layers utilize **weight sharing** to reduce the number of parameters compared to linear layers.
    *   **Normalization:** Techniques like **Batch Norm**, **Layer Norm**, and **Group Norm** are used to stabilize training.
    *   **Activation Functions:** These non-linear functions include **ReLU**, **ReLU6**, **Leaky ReLU**, **Sigmoid**, and **Hard Swish**.

### **II. Efficiency Metrics: Memory**
Measuring memory cost is critical for deploying models on resource-constrained devices:
*   **#Parameters:** The total count of elements in a network's weight tensors.
*   **Model Size:** Calculated as the number of parameters multiplied by the **bit width** (e.g., 32-bit float vs. 8-bit integer).
*   **#Activations:** Often the primary **memory bottleneck** during inference, particularly for CNNs. Peak activation memory is roughly the sum of the input and output activation sizes for a given layer.

### **III. Efficiency Metrics: Computation and Performance**
*   **MACs and FLOPs:** A **Multiply-Accumulate (MAC)** operation consists of one multiplication and one addition ($a \leftarrow a + b \cdot c$). One MAC operation is equivalent to **two Floating Point Operations (FLOPs)**.
*   **Latency vs. Throughput:** 
    *   **Latency** measures the delay for a specific task (e.g., milliseconds per image). It is generally limited by either the time taken for computation or the time taken for memory access.
    *   **Throughput** measures the rate of data processing (e.g., images per second). Importantly, higher throughput does not always translate to lower latency.
*   **Energy Consumption:** Data movement is the most significant contributor to energy cost. For instance, accessing **DRAM** is roughly **200 times** more energy-intensive than a 32-bit integer multiplication.

## References
- Source: https://notebooklm.google.com/notebook/39b7bd5c-37ef-451c-83dc-98b62ec8dc04
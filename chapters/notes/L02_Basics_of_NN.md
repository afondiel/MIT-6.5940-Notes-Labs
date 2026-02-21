# L2: Basics of Deep Learning

## Quick Reference

|Item|Reference|
|---|---|
| Slides | [View Slides](https://drive.google.com/drive/folders/1A3P6IBuS8wCzLlpdRiQBO9b1uoK3pnPf?usp=sharing)|
| Video | [EfficientML.ai Lecture 2 - Basics of Neural Networks](https://www.youtube.com/watch?v=ieg0RJb7TeI)  |
|Lab| [L02_NN_Basics.ipynb](../../lab/notebooks/) |
|Professor|[Song Han](https://github.com/songhan)|

## **1\. Review of Neural Network Fundamentals**

This lecture first reviews the core components and operations of Deep Neural Networks (DNNs) to establish a baseline for understanding where efficiency can be gained.

### **A. Core Terminology**

- **Neuron/Node:** The fundamental unit, performing an affine transformation ( $W^T X + b$ ) followed by a non-linear activation function.  
- **Synapses:** The connections between neurons, storing the weights ($W$) and biases ($b$).  
- **Parameters/Weights:** The trainable values in the network (e.g., $W$ and $b$).

### **B. Fundamental Building Blocks**

The efficiency of a network is determined by its architecture, which is built from basic layers:

1. **Fully-Connected Layer / Linear Layer:** Every input is connected to every output. Highly parameterized, leading to high memory cost.  
2. **Convolutional Layer (Conv):** Uses a small kernel (filter) to apply weights across a local region of the input feature map, leading to weight sharing and reduced parameters compared to FC layers.  
   * **Grouped Convolution:** Divides channels into groups, reducing computation and parameters.  
   * **Depthwise Convolution:** A special case where the number of groups equals the number of input channels ($g = C_{in}$). This is extremely efficient, often used in MobileNet.  
3. **Pooling Layers (Max/Avg):** Reduce spatial resolution (H x W) and computational load.  
4. **Normalization Layers (Batch Norm, Layer Norm):** Stabilize training.  
5. **Transformer Blocks:** Use Multi-Head Attention (MHA) and Feed-Forward Networks (FFN). Attention layers are a major computational and memory bottleneck, especially in LLMs.

## **2\. Essential Efficiency Metrics (Quantifying Cost)**

The core focus of efficiency in this course is on defining and calculating the metrics that determine a model's computational and memory cost.

### **A. Computational Cost Metrics**

These metrics measure the theoretical amount of work performed by the model during a forward pass.

| Metric | Full Name | Definition | Relevance |
| :---- | :---- | :---- | :---- |
| **MACs** | Multiply-Accumulate Operations | A single operation: $A \times B + C$. The most common operation in neural networks. | **Hardware-Friendly:** Often used by hardware architects as a single cycle operation. |
| **FLOPs** | Floating-Point Operations | Any single floating-point math operation (+, \-, $\times$, /). Typically, **1 MAC** $\approx$ **2 FLOPs** (1 multiplication \+ 1 addition). | **Algorithm-Friendly:** Standardized metric for computational complexity, regardless of hardware. |
| **GFLOPs** | Giga-FLOPs ($10^9$ FLOPs) | Common unit for large models. | Practical measure for comparing models like ResNet-50 ($\approx 4$ GFLOPs). |

#### **Calculating MACs/FLOPs**

1. Fully-Connected Layer:  
   $$
   \text{MACs} = 
   \text{Batch Size} \times 
   \text{Input Size} \times 
   \text{Output Size}
   $$  
2. Convolution Layer:  
   $$
   \text{MACs} \approx 
   \text{Output H} \times 
   \text{Output W} \times 
   \text{Kernel H} \times 
   \text{Kernel W} \times 
   \text{Input C} \times 
   \text{Output C}
   $$

**Key Insight:** For standard convolutional layers, the computational cost (MACs/FLOPs) scales quadratically with the spatial resolution ($H \times W$) and the number of channels ($C_{in} \times C_{out}$).

### **B. Memory Cost Metrics**

These metrics measure the resources required to store the model and its intermediate results.

| Metric | Definition | Relevance |
| :---- | :---- | :---- |
| **\#Parameters** | Total number of weights and biases in the model. | **Storage Cost:** Directly determines the model's disk size and VRAM usage for weights. |
| **Model Size** | The physical size of the model (e.g., 1 Parameter (FP32) \= 4 Bytes). | Determines the time required to load the model (Memory Bandwidth). |
| **Peak \#Activations** | Maximum memory required to store the intermediate feature maps during a forward pass. | **RAM/VRAM Bottleneck:** Critical for inference on memory-constrained devices (e.g., TinyML) and for large batch training. |

### **C. Time and Performance Metrics**

These metrics measure the real-world execution speed.

| Metric | Definition | Importance |
| :---- | :---- | :---- |
| **Latency** | The wall-clock time required for a single inference (input to output). | **User Experience:** Critical for real-time applications (e.g., autonomous driving, video processing). |
| **Throughput (FPS/TPS)** | The number of inferences (Frames/Tokens) processed per second. | **Server Efficiency:** Critical for maximizing the serving capacity of a cloud GPU cluster. |

## **3\. The Efficiency Bottleneck: Memory Access**

Recalling L1's core principle, the lecture emphasizes that the true bottleneck for inference speed is often **Memory Access Cost**, not just the computational FLOPs.

* A model with low FLOPs but poor memory access patterns can be slower than a model with high FLOPs but excellent memory locality (e.g., a **Dense layer vs. a Grouped Conv**).  
* **Data Types:** The memory footprint is directly proportional to the numerical precision used (FP32 \> FP16 \> INT8). This motivates the entire study of **Quantization** (L5, L6).

## **4\. Lab 0: PyTorch and Efficiency Measurement**

* **Lab Goal:** Introduce PyTorch, basic neural network implementation, and hands-on calculation of the introduced efficiency metrics.  
* **Key Skills:** Students should be able to define, calculate, and measure **FLOPs/MACs**, **Parameters**, and **Latency** for simple models. This forms the baseline for all subsequent efficiency labs.

## References

- EfficientML.ai Course | 2023 Fall | MIT 6.5940: [Complete course video series](https://youtube.com/playlist?list=PL80kAHvQbh-pT4lCkDT53zT8DKmhE0idB&si=Uu00N0zKopEixhw3)
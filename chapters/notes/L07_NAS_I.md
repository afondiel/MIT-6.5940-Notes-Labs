# Lecture 7: Neural Architecture Search (Part I)
## Quick Reference

|Item|Reference|
|---|---|
| Slides | [Slides](https://drive.google.com/drive/folders/1A3P6IBuS8wCzLlpdRiQBO9b1uoK3pnPf?usp=sharing)|
| Video | [EfficientML.ai Lecture 7 - Neural Architecture Search (Part I)](https://www.youtube.com/watch?v=W84MKJSg90A)  |
|Lab| [Lab3-NAS.ipynb](../../lab/notebooks/Lab3.ipynb) |
|Professor|[Song Han](https://github.com/songhan)|


## **1\. Introduction and Motivation for NAS**

**Neural Architecture Search (NAS)** is the process of automating the design of neural network architectures. It moves from manually engineered architectures (like ResNet or VGG) to automatically discovered ones (like MobileNetV3 or EfficientNet).

### **A. The Challenge of Manual Design**

* **Complexity:** Designing an efficient and accurate network is extremely time-consuming, requires deep expertise, and involves numerous decisions (layer types, kernel sizes, skip connections, activation functions, etc.).  
* **The Design Trade-off:** The goal is to find the **Pareto Optimal Frontier**—the best set of models that offer the maximum accuracy for a given latency or computational budget (FLOPs). Manual search rarely finds this optimal point.  
* **Hardware Awareness:** An architecture optimal for a GPU may be sub-optimal for a specialized mobile NPU (Neural Processing Unit). NAS can be made **hardware-aware** to find the best model for a specific device.

### **B. The Three Pillars of a NAS System**

Any NAS framework consists of three mandatory components that define the search process:

1. **Search Space:** Defines the set of possible neural network architectures that can be generated.  
2. **Search Strategy:** Defines the algorithm used to explore the search space efficiently.  
3. **Performance Estimation Strategy:** Defines how the model's performance (e.g., accuracy and latency) is quickly evaluated without full, expensive training.

## **2\. Search Space Definition**

The search space defines the structural possibilities of the network. A small space is cheap but limits innovation; a large space is expressive but impossible to search exhaustively.

### **A. Hierarchical Search Spaces**

To balance expressiveness and search cost, NAS often uses a hierarchical design, where the network is built by stacking small, optimized "cells."

1. **Layer-based Search:** The entire network is designed layer by layer. The space is vast (e.g., millions of possible layer combinations), making the search computationally prohibitive. (Used in early NASNet/AutoML work).  
2. **Cell-based Search:** The entire network is a repeated stack of two core, learned cells:  
   * **Normal Cell:** The main building block used to process features while maintaining the spatial resolution.  
   * **Reduction Cell:** Used to downsample the spatial resolution (e.g., for stride 2\) while increasing the channel count.  
   * **Benefit:** Greatly reduces the search space size by optimizing a small, transferable primitive (the cell) which is then stacked many times.

### **B. Operation Set and Connections**

Within a cell, the architecture is defined by:

* **Operations:** The allowed functions (e.g., $3 \times 3$ Conv, $5 \times 5$ Max Pool, $1 \times 1$ Conv, Depthwise Conv, Identity).  
* **Connections:** How the inputs of one operation connect to the outputs of others (e.g., skip connections).

## **3\. Search Strategy: Exploring the Space**

The search strategy determines the algorithm used to select the next promising architecture to evaluate.

### **A. Reinforcement Learning (RL)**

* **Mechanism:** An RL **Controller** (often an RNN) is trained to generate the string of tokens that defines the network architecture.  
  * The **Action** is generating a layer/cell design.  
  * The **Reward** is the validation accuracy of the sampled network after training.  
* **Drawback:** Very slow. Requires training (or partially training) thousands of models, often consuming thousands of GPU days.

### **B. Evolutionary Search (EA)**

* **Mechanism:** Starts with a random population of network architectures. At each iteration, the best architectures are selected, **mutated** (e.g., change a kernel size, add a skip connection), and the resulting offspring are evaluated.  
* **Benefit:** Can discover novel, non-intuitive connections and is often more robust than RL in exploring complex spaces.

### **C. Differentiable Architecture Search (DARTS)**

DARTS is the most efficient and dominant strategy today, reducing the search cost from thousands of GPU days to hours.

* **Core Idea:** Make the search space and the architecture selection process **fully differentiable**.  
* **Mechanism (Supernet):** All possible operations and connections in the search space are modeled simultaneously as a **Supernet**.  
  * The search process learns both the **weight parameters** ($W$) *and* the **architecture parameters** ($\alpha$).  
  * $W$ is learned via standard backpropagation (like training a regular model).  
  * $\alpha$ (which determines *which* operation to select) is also learned via backpropagation, often using a validation set.  
* **Result:** The search becomes a continuous optimization problem, eliminating the need to train individual sub-networks.

## **4\. Performance Estimation: Speeding up Evaluation**

Even with efficient search strategies, fully training a candidate architecture takes too long. Performance Estimation strategies rapidly estimate a model's quality.

### **A. Weight Sharing (The Key to Efficiency)**

* **Concept:** Instead of training each candidate sub-network from scratch, all candidate sub-networks **share their weights** from a single, large **Supernet**.  
* **Mechanism:** The Supernet is trained once. When a specific sub-network (architecture) is sampled, it inherits the relevant weights from the Supernet and is quickly evaluated, often without further training.  
* **Benefits:** Reduces the training cost for the *entire search space* to the cost of training just one Supernet. This is the foundation of most modern, efficient NAS methods.

### **B. Hardware-Aware Metrics**

To truly find the *efficient* frontier, the reward function must move beyond just accuracy and incorporate hardware costs.

$$
\text{Reward} = \text{Accuracy} \times \left[ \frac{\text{Latency}_{\text{target}}}{\text{Latency}_{\text{measured}}} \right]^k
$$

* **Goal:** Penalize models that are slow on the target hardware.  
* **Measured Latency:** The latency is often measured directly on the target hardware (e.g., a phone, a Jetson board, or a microcontroller) using a dedicated profiling tool, ensuring the search finds architectures that are genuinely fast in deployment.

## **5\. Summary and Impact**

NAS has been critical in pushing the efficiency frontier, resulting in architectures that perform far better than their manually designed counterparts.

* **MobileNetV3 and EfficientNet:** Key examples of architectures discovered through AutoML/NAS, demonstrating superior trade-offs between accuracy and computational cost (FLOPs/latency).  
* **MicroNets:** NAS used specifically to design tiny networks that fit into the **Kilobytes of memory** constraints of microcontrollers, a foundational concept for TinyML.

## References

- EfficientML.ai Course | 2023 Fall | MIT 6.5940: [ Complete course video series ](https://youtube.com/playlist?list=PL80kAHvQbh-pT4lCkDT53zT8DKmhE0idB&si=Uu00N0zKopEixhw3).
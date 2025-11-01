# Lecture 8: Neural Architecture Search (Part II)

## Quick Reference

|Item|Reference|
|---|---|
| Slides | [View Slides](https://drive.google.com/drive/folders/1A3P6IBuS8wCzLlpdRiQBO9b1uoK3pnPf?usp=sharing)|
| Video | [EfficientML.ai Lecture 8 - Neural Architecture Search (Part II)](https://www.youtube.com/watch?v=MOMc8KkGwCc)  |
|Lab| [Lab3.ipynb](../../lab/notebooks/Lab3.ipynb) |
|Professor|[Song Han](https://github.com/songhan)|


## Overview

This lecture dives into the **advanced** and practical aspects of Neural Architecture Search (NAS), moving beyond basic search strategies to focus on efficiency, hardware-awareness, and co-design.

## **1. Efficient Neural Architecture Search (NAS)**

Traditional NAS is computationally expensive, often requiring thousands of GPU hours. Efficient NAS techniques aim to drastically reduce this cost.

### **A. Weight Sharing (One-Shot NAS)**

- **Concept:** Instead of training thousands of individual architectures, a **SuperNet** (or one-shot model) is created, which encompasses all possible architectures in the search space. 
- **Mechanism:** All sub-networks within the SuperNet share weights. The SuperNet is trained once, and then the NAS algorithm searches for the best path (sub-network) within the SuperNet without retraining weights. 

- **Example:** **DARTS (Differentiable Architecture Search)** is a prominent weight-sharing method that uses continuous relaxation and gradient-based optimization to update both the shared weights and the architecture parameters. 

### B. Pruning-Based NAS

- **Concept:** Start with a large, over-parameterized SuperNet and prune away the less effective components to find an optimal compact architecture. 

- **Once-For-All (OFA) Network:** A single, fully trained SuperNet that supports *all* sub-networks (different depths, widths, and kernel sizes). This allows the architecture to be specialized for different deployment platforms (like Mobile, GPU, CPU) *without* any retraining. 

## 2. Hardware-Aware NAS (HW-NAS)

NAS traditionally optimizes for accuracy only. HW-NAS incorporates hardware constraints (like latency, power consumption, or memory footprint) directly into the search objective.

- **Motivation:** An architecture that is accurate on a GPU might be extremely slow on a mobile phone due to different memory access and computation patterns. HW-NAS ensures the found architecture is optimal for a specific target hardware.  
- **Search Objective:** The optimization criterion is typically a weighted combination:

$$
\text{Maximize} \quad
\text{Accuracy} - \alpha \times 
\text{Latency}
$$

where $\alpha$ is a hyperparameter balancing the two metrics.

- **Latency Prediction:** To avoid running actual inference (which is slow) for every candidate architecture, a **Latency Predictor** is trained. This predictor is usually a lightweight machine learning model that takes the architectural parameters (e.g., number of layers, kernel sizes) as input and quickly outputs the estimated latency on the target device.

## 3. Neural-Hardware Architecture Co-Search

This is the most advanced form of NAS, where the neural network architecture and the hardware accelerator architecture are designed *simultaneously* and *interdependently*.

- **Challenge:** Hardware accelerators (like ASICs or FPGAs) are highly specialized. The optimal hardware design depends on the neural network (NN) structure, and the optimal NN structure depends on the hardware's capabilities.  
- **Co-Search Loop:** A joint optimization is performed where the search space includes both NN parameters (layers, connections) and hardware parameters (bit-width, memory hierarchy, processing element count). 

- **Goal:** Find a pair of $(\text{NN}, \text{Hardware})$ that achieves the highest performance (e.g., accuracy / latency) for the given budget.

## 4. Zero-Shot NAS (Future Directions)

Zero-Shot NAS explores ways to evaluate the potential performance of a neural network architecture **without any training whatsoever**.

-  **Concept:** Use statistical proxies or analytical metrics (like number of linear regions, or measures of network capacity) that correlate strongly with the final trained accuracy to rank architectures instantly.

- **Benefit:** If successful, this would entirely eliminate the need for training (even SuperNet training), making NAS extremely fast.  
- **Example:** Metrics based on the Hessian or Jacobian of the network, which measure the complexity or "trainability" of the architecture.


## References

- EfficientML.ai Course | 2023 Fall | MIT 6.5940: [ Complete course video series ](https://youtube.com/playlist?list=PL80kAHvQbh-pT4lCkDT53zT8DKmhE0idB&si=Uu00N0zKopEixhw3).
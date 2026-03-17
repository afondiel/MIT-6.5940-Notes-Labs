# **Lesson 7: Neural Architecture Search (Part I)** 

introduces Neural Architecture Search (NAS), an automated technique for designing neural network architectures that balances the trade-offs between efficiency (storage, latency, energy) and accuracy.

### **I. Transition to Automated Design**
The core motivation for NAS is moving from **Manual Architecture Search**, which relies on human expertise and trial-and-error, to **Automatic Architecture Search**, which uses machine learning to find the optimal architecture within a defined set of possibilities. The goal is to maximize an objective function, typically a combination of accuracy and hardware efficiency.

### **II. NAS Components**
A standard NAS framework consists of three main components:
1.  **Search Space ($\mathcal{A}$):** The set of all candidate architectures the algorithm can choose from.
2.  **Search Strategy:** The method used to explore the search space.
3.  **Performance Estimation Strategy:** How the algorithm predicts or measures the quality (e.g., accuracy) of a candidate architecture.

### **III. Search Space Design**
Search spaces are generally categorized into two levels of granularity:
*   **Cell-level Search Space:** Instead of designing the whole network, the algorithm designs small "cells" (e.g., **Normal Cells** and **Reduction Cells**) that are stacked to form the final architecture. In models like **NASNet**, an RNN controller recursively selects inputs and transformation operations (like convolution or pooling) to build these cells.
*   **Network-level Search Space:** This focuses on the **topology and connections** of the entire network. For example, **Auto-DeepLab** uses a trellis-like search space where different paths through nodes correspond to different network architectures for image segmentation.
*   **TinyNAS:** Specifically designed for **TinyML**, this approach uses automated search space optimization to ensure candidate models can fit within extreme memory and storage constraints.

### **IV. Search Strategies**
The lecture covers several strategies for exploring the search space:
*   **Grid Search:** The traditional method for hyperparameter optimization, often used for compound scaling of depth, width, and resolution (e.g., **EfficientNet**).
*   **Reinforcement Learning (RL):** A controller (typically an RNN) samples an architecture, which is then trained to provide a reward (accuracy) used to update the controller's policy.
*   **Gradient Descent (Differentiable Search):** Techniques like **DARTS** use **continuous relaxation** to make the choice between operations differentiable. By using a softmax over all possible operations, the algorithm can jointly learn architecture parameters ($\alpha$) and network weights ($w$) using gradient descent, significantly reducing search time.
*   **Other Methods:** These include **Random Search** and **Evolutionary Search**.

### **V. Summary of Primitive Operations**
The lesson also reviews the fundamental building blocks used in these search spaces, such as standard convolutions, grouped convolutions, depthwise convolutions, and **Multi-Head Self-Attention (MHSA)** for Transformers. Understanding these operations is essential for defining the operations available to the NAS algorithm.


## References
- Source: https://notebooklm.google.com/notebook/39b7bd5c-37ef-451c-83dc-98b62ec8dc04
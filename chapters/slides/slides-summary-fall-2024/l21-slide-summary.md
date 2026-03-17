# **Lesson 21: On-Device Training and Transfer Learning** 

explores the techniques and system support required to enable deep learning models to learn and adapt directly on resource-constrained edge devices while preserving user privacy.

### **I. Motivation: Customization and Privacy**
*   **Benefits:** On-device learning allows AI systems to **continually adapt** to new data collected from sensors (customization) and ensures that sensitive user data (like private code or enterprise data) **never leaves the local device**.
*   **Federated Learning:** In this paradigm, devices perform local training and only share **gradients or weights** with a central server, which averages them and sends the updated model back to the devices.
*   **The Safety of Gradients:** Sharing gradients is not inherently safe; **Deep Leakage from Gradients (DLG)** demonstrates that raw training data can be reconstructed from publicly shared gradients. Techniques like **Deep Gradient Compression (DGC)** can help protect privacy by obfuscating gradients through local accumulation.

### **II. Challenges: The Memory Bottleneck**
*   **Memory Constraints:** On-device training is significantly more challenging than inference because edge devices (like microcontrollers with **320KB SRAM**) have memory sizes orders of magnitude smaller than cloud servers.
*   **Activation Storage:** Training memory is much larger than inference memory because the **backward pass** requires storing **intermediate activations** to calculate weight gradients, which can exceed device limits by many times.

### **III. Algorithmic Optimizations for Efficient Training**
*   **Tiny Transfer Learning (TinyTL):** Reduces the activation memory footprint by focusing on updating only **biases** or small "lite-residual" modules instead of the full weight tensors, as updating biases does not require storing activations.
*   **Sparse Back-propagation (SparseBP):** Achieves higher accuracy with **4.5x to 7.5x less extra memory** by updating only a specialized subset of layers or tensors during training.
*   **Quantized Training and QAS:** Standard Quantization-Aware Training (fake quantization) fails to save memory because intermediate tensors remain in FP32. Real quantized training uses **integer tensors** to save memory, and **Quantization-Aware Scaling (QAS)** is used to maintain accuracy comparable to full-precision training.

### **IV. System Support: PockEngine**
*   **Algorithm-System Co-Design:** **PockEngine** is a framework designed to translate these algorithmic improvements into actual hardware savings.
*   **Compile-Time Autodiff:** Unlike conventional frameworks that perform auto-differentiation at runtime, PockEngine moves the workload to **compile-time**. This minimizes runtime overhead and enables extensive **graph-level optimizations** such as operator reordering, constant folding, and sparse updates.
*   **Performance:** PockEngine demonstrates significant speedups—ranging from **21x to 23x faster** than TensorFlow-Lite—across diverse platforms including ARM CPUs (Raspberry Pi), edge GPUs (Jetson Orin), and Apple M-series chips.

## References
- Source: https://notebooklm.google.com/notebook/39b7bd5c-37ef-451c-83dc-98b62ec8dc04
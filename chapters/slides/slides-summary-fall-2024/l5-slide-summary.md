# **Lesson 5: Quantization (Part I)** 

introduces the fundamental concepts of neural network quantization, focusing on reducing the precision of weights and activations to improve hardware efficiency, energy consumption, and storage.

### **I. Motivation and Definition**
*   **Definition:** Quantization is the process of mapping continuous or large sets of values to a limited, discrete set. The difference between an original value and its quantized version is known as **quantization error**.
*   **Energy Efficiency:** Lower bit-width operations are significantly cheaper; for example, an **8-bit integer multiplication** consumes roughly **18.5x less energy** than a 32-bit floating-point multiplication.
*   **Data Movement:** Moving data from **DRAM** is the most expensive operation (e.g., 200 pJ for a 32-bit access), making model compression vital for energy-constrained devices.

### **II. Numeric Data Types**
Modern systems use various formats to represent numbers, each balancing **range** (governed by the exponent width) and **precision** (governed by the fraction/mantissa width):
*   **FP32 (Single Precision):** 1 sign bit, 8 exponent bits, and 23 fraction bits.
*   **FP16 (Half Precision):** 5 exponent bits and 10 fraction bits.
*   **BF16 (Brain Float):** 8 exponent bits (same range as FP32) and 7 fraction bits.
*   **FP8/FP4:** New low-precision formats used in advanced hardware like **NVIDIA Blackwell** for higher throughput.

### **III. K-Means-based Quantization**
Commonly used in the **Deep Compression** framework, this method clusters weights into groups and represents them using the group's centroid.
*   **Storage:** Weights are stored as **low-bit integer indices** (e.g., 2-bit) pointing to a **floating-point codebook** of centroids.
*   **Computation:** While storage is reduced, actual computation typically still uses **floating-point arithmetic** because weights must be de-quantized during the forward pass.

### **IV. Linear Quantization**
Linear quantization uses an **affine mapping** to transform integers to real numbers using the formula **$r = S(q - Z)$**, where $S$ is the scale and $Z$ is the zero point.
*   **Storage & Computation:** Both weights and activations are stored as integers, and operations can be performed using **integer arithmetic**, which is faster and more energy-efficient.
*   **Scale ($S$):** Calculated as $\frac{r_{max} - r_{min}}{q_{max} - q_{min}}$ to map the floating-point range to the integer range.
*   **Zero Point ($Z$):** An integer that ensures the real value 0.0 is exactly representable, calculated as $round(q_{min} - \frac{r_{min}}{S})$.
*   **Symmetric Quantization:** A simplified version where the zero point is forced to be **$Z = 0$**, and the range is symmetric around zero.

### **V. The Deep Compression Pipeline**
Lesson 5 concludes by integrating quantization into a broader optimization strategy.
1.  **Pruning:** Reduces the number of weights by roughly 10x.
2.  **Quantization:** Reduces the bits used per weight, further compressing the model (up to 27x–31x combined).
3.  **Huffman Coding:** Exploits the biased distribution of remaining values to achieve final compression ratios of **35x to 49x** without losing accuracy.
4.  **Results:** Applying this pipeline to already compact models like **SqueezeNet** can achieve up to **510x reduction** in size.

## References
- Source: https://notebooklm.google.com/notebook/39b7bd5c-37ef-451c-83dc-98b62ec8dc04
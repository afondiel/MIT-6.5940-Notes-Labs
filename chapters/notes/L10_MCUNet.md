# Lecture 10: MCUNet: TinyML on Microcontrollers

## Quick Reference

|Item|Reference|
|---|---|
| Slides | [View Slides](https://drive.google.com/drive/folders/1A3P6IBuS8wCzLlpdRiQBO9b1uoK3pnPf?usp=sharing)|
| Video | [MCUNet: TinyML on Microcontrollers ](https://www.youtube.com/watch?v=zJ3ZDZXD_zw)  |
|Lab| -- |
|Professor|[Song Han](https://github.com/songhan)|


## **1. The Computer Architecture Context**

To design efficient ML, we must understand the fundamental bottlenecks in modern computing, particularly the distinction between **computation** and **memory access**.

### **A. The Von Neumann Bottleneck**

* **Problem:** Traditional computer architectures (Von Neumann) use separate units for the processor (CPU/ALU) and memory (DRAM). All data must be fetched from memory and moved to the processor for computation.  
* **Energy Cost:** Moving data is vastly more expensive than computing it.  
  * **Cost of 32-bit Floating-Point (FP) Operation:** $\approx 0.9$ picojoules (pJ).  
  * **Cost of Moving 32-bit Data (Off-Chip DRAM):** $\approx 100-1000$ pJ.  
* **Implication:** **Efficiency is bottlenecked by data movement.** Architectures must minimize data movement, which motivates specialized AI accelerators.

### **B. Specialized Hardware for ML**

General-purpose CPUs and GPUs are not optimal for the repetitive, high-throughput Multiply-Accumulate (MAC) operations central to deep learning.

| Hardware Type | Design Goal | Key Operation | Energy Efficiency (TOPS/W) |
| :---- | :---- | :---- | :---- |
| **CPU** | Flexibility, Low Latency (single thread) | Single-Thread Performance, Branching | Low ($\sim 0.1$) |
| **GPU** | High Parallelism (SIMT), High Throughput | Matrix Multiplication (32-bit FP) | Medium ($\sim 1-10$) |
| **ASIC / Accelerator (e.g., TPU)** | Specificity, Max Efficiency | Massive MAC Array, Low-Bit/Integer Arithmetic | High ($\sim 10-100$) |

## **2\. Key Architectural Components of AI Accelerators**

AI accelerators (ASICs like Google TPU or specialized IP) are designed around maximizing MAC throughput and minimizing memory access.

### **A. Processing Element (PE) Array**

* The heart of the accelerator is the **Systolic Array** (or PE Array), a grid of interconnected Processing Elements (PEs).  
* **Function:** PEs perform the MAC operations. Weights and input data flow rhythmically across the array, allowing data reuse and eliminating the need for constant memory fetches.  
* **Data Reuse:** Maximizing reuse of data (input activation, weight, output accumulation) within the PE array is the primary strategy for power efficiency.

### **B. Memory Hierarchy**

Accelerators rely on a carefully managed memory hierarchy to keep data close to the PEs.

1. **On-Chip Memory (SRAM):** Small, extremely fast, and highly energy-efficient memory located directly on the chip (e.g., Buffers, Cache). Data in SRAM is $\sim 100\times$ faster/cheaper to access than off-chip DRAM.  
2. **Off-Chip Memory (DRAM):** Large capacity, but slow and power-hungry (e.g., HBM in modern GPUs/TPUs).

### **C. Dataflow (How Data is Used)**

The dataflow defines how input activations, weights, and partial sums are moved through the PE array and memory hierarchy. The goal is to maximize **Data Reuse**.

| Dataflow Type | Principle | Reuse Focus | Hardware Implication |
| :---- | :---- | :---- | :---- |
| **Weight Stationary (WS)** | Load weights once into the PE, stream activations through. | Maximizes **Weight Reuse**. | Effective for Convolutional Layers. |
| **Output Stationary (OS)** | Fix the partial sum (output feature map) in the PE, stream weights and activations. | Maximizes **Output Reuse**. | Good for large batch sizes. |

## **3\. Optimizing for Low-Precision (Quantization Hardware)**

Hardware achieves massive efficiency gains by shifting from 32-bit floating-point (FP32) to low-bit integer arithmetic (INT8, INT4).

### **A. Integer Arithmetic Benefits**

* **Size:** INT8 weights require $4\times$ less memory bandwidth than FP32.  
* **Power:** Integer adders and multipliers are significantly smaller and consume much less power than their FP counterparts.  
* **Tensor Cores:** Dedicated units in modern GPUs (NVIDIA) and specialized ASICs (TPU, Qualcomm) are optimized for **low-precision matrix multiplication**, allowing for higher effective throughput (TOPS) when using INT8 or INT4.

### **B. Handling Sparsity**

As covered in L4, standard dense hardware cannot benefit from unstructured sparsity.

* **Custom Indexing:** To gain speedup from sparse weights, the hardware must include dedicated logic to skip zero-MACs and manage the complex indexing/addressing of non-zero data (e.g., specialized **EIE (Efficient Inference Engine)** architecture).

## **4\. Introduction to TinyML Hardware**

The constraints of TinyML (milliwatts power, Kilobytes memory) require extreme architectural efficiency.

* **Microcontrollers (MCUs):** Typically use low-power CPUs (like ARM Cortex-M series) that lack large SIMD units or complex systolic arrays.  
* **Solution: Micro-Architectural Co-Design:** TinyML efficiency is achieved by co-designing the network architecture (e.g., MobileNet) and the inference runtime (e.g., **TinyEngine** from MIT) to fit within the memory and compute limits of the MCU.

This lecture provided the foundational hardware knowledge needed to understand the design constraints for efficient ML deployment. We can now proceed to **Lecture 11: TinyML Systems**, which focuses specifically on the challenges and solutions for extreme edge computing.
## References

- EfficientML.ai Course | 2023 Fall | MIT 6.5940: [ Complete course video series ](https://youtube.com/playlist?list=PL80kAHvQbh-pT4lCkDT53zT8DKmhE0idB&si=Uu00N0zKopEixhw3).
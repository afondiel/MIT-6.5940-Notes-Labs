# **EfficientML.ai Lecture 23: Noise Robust Quantum ML**

## Overview

This lecture addresses the critical challenge of noise in **Noisy Intermediate-Scale Quantum (NISQ)** devices and its impact on Quantum Machine Learning (QML) performance.

## Lecture Materials

|Item|Reference|
|---|---|
|Slides|[View Slides](https://drive.google.com/file/d/17JORXimIgO3DKnRfeLuIgVdw_mp2SNWP/view?usp=sharing)|
|Lab| -- |
|Professor|Hanrui Wang|
|Course|MIT 6.5940 (Fall 2023): [TinyML and Efficient Deep Learning Computing](https://www.youtube.com/playlist?list=PL80kAHvQbh-pT4lCkDT53zT8DKmhE0idB)  |

## **1\. The Challenge of Quantum Noise**

The lecture begins by outlining why quantum noise is a first-order concern for QML, especially for algorithms like **Variational Quantum Eigensolver (VQE)** and **Quantum Approximate Optimization Algorithm (QAOA)**, which rely on iterative optimization \[2:20\].

* **Noise Sources:** Quantum noise arises from various sources, including decoherence, gate imperfections, and measurement errors \[3:00\].  
* **The Barren Plateau Problem:** Noise exacerbates the barren plateau problem, where the gradients of the cost function vanish exponentially with the number of qubits, making training QML models impossible \[5:45\].

## **2\. Techniques for Noise Robustness**

The core of the lecture focuses on two major approaches to mitigating noise: Quantum Error Mitigation (QEM) and noise-aware circuit design.

### **A. Quantum Error Mitigation (QEM)**

QEM techniques aim to statistically suppress noise effects without requiring the massive overhead of full quantum error correction.

* **Zero-Noise Extrapolation (ZNE):** This technique scales the noise level of the circuit (e.g., by repeating gates) and then uses the results at different noise levels to mathematically extrapolate the result back to the theoretical zero-noise limit \[19:40\].  
* **Probabilistic Error Cancellation (PEC):** A method that learns the inverse of the quantum channel (the noise model) and uses classical post-processing to probabilistically sample the noiseless output from the noisy measurements \[16:00\].  
* **Learning-Based Mitigation (LBM):** This approach uses a classical machine learning model (e.g., a neural network) to predict the expected noiseless outcome based on the noisy measurement data, effectively learning to compensate for the hardware noise characteristics \[30:00\].

### **B. Noise-Aware Circuit Design and Pruning**

Classical machine learning efficiency techniques, such as pruning and hardware-aware design, are adapted to create inherently more noise-robust quantum circuits.

* **Quantum Circuit Pruning:** This involves strategically removing or simplifying parts of the parametrized quantum circuit (the ansatz) to make it shallower, reducing the overall exposure to gate errors and decoherence time \[45:00\].  
* **Noise-Aware Compilation:** This involves optimizing the mapping of the logical quantum circuit to the physical qubits on the hardware, aiming to minimize the use of noisy gates (like two-qubit CNOTs) and avoiding qubit connections with high error rates \[40:00\].

## **3\. Quantum Architecture Search (QAS)**

The lecture explores the automated design of quantum circuits, where an algorithm searches for the optimal circuit structure (ansatz) that is both expressive for the task and resilient to hardware noise \[55:00\]. QAS integrates the noise model into the objective function to find a trade-off between model capacity and noise-induced error, ultimately resulting in more efficient and robust QML models for NISQ devices.

## References

- [Lecture YT Video](https://www.youtube.com/watch?v=kCTzlodCZII)
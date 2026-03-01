# **Lecture 23: Noise Robust Quantum ML**

## Quick Reference

|Item|Reference|
|---|---|
|Slides| [Slides](https://drive.google.com/drive/folders/1A3P6IBuS8wCzLlpdRiQBO9b1uoK3pnPf?usp=sharing)|
|Course| [Noise Robust Quantum ML](https://www.youtube.com/watch?v=kCTzlodCZII)  |
|Lab| -- |
|Professor|Hanrui Wang|

## Overview

This lecture addresses the critical challenge of noise in **Noisy Intermediate-Scale Quantum (NISQ)** devices and its impact on Quantum Machine Learning (QML) performance.


## **1\. The Challenge of Quantum Noise**

The lecture begins by outlining why quantum noise is a first-order concern for QML, especially for algorithms like **Variational Quantum Eigensolver (VQE)** and **Quantum Approximate Optimization Algorithm (QAOA)**, which rely on iterative optimization.

* **Noise Sources:** Quantum noise arises from various sources, including decoherence, gate imperfections, and measurement errors.  
* **The Barren Plateau Problem:** Noise exacerbates the barren plateau problem, where the gradients of the cost function vanish exponentially with the number of qubits, making training QML models impossible.

## **2\. Techniques for Noise Robustness**

The core of the lecture focuses on two major approaches to mitigating noise: Quantum Error Mitigation (QEM) and noise-aware circuit design.

### **A. Quantum Error Mitigation (QEM)**

QEM techniques aim to statistically suppress noise effects without requiring the massive overhead of full quantum error correction.

* **Zero-Noise Extrapolation (ZNE):** This technique scales the noise level of the circuit (e.g., by repeating gates) and then uses the results at different noise levels to mathematically extrapolate the result back to the theoretical zero-noise limit.  
* **Probabilistic Error Cancellation (PEC):** A method that learns the inverse of the quantum channel (the noise model) and uses classical post-processing to probabilistically sample the noiseless output from the noisy measurements.  
* **Learning-Based Mitigation (LBM):** This approach uses a classical machine learning model (e.g., a neural network) to predict the expected noiseless outcome based on the noisy measurement data, effectively learning to compensate for the hardware noise characteristics.

### **B. Noise-Aware Circuit Design and Pruning**

Classical machine learning efficiency techniques, such as pruning and hardware-aware design, are adapted to create inherently more noise-robust quantum circuits.

* **Quantum Circuit Pruning:** This involves strategically removing or simplifying parts of the parametrized quantum circuit (the ansatz) to make it shallower, reducing the overall exposure to gate errors and decoherence time.  
* **Noise-Aware Compilation:** This involves optimizing the mapping of the logical quantum circuit to the physical qubits on the hardware, aiming to minimize the use of noisy gates (like two-qubit CNOTs) and avoiding qubit connections with high error rates.

## **3\. Quantum Architecture Search (QAS)**

The lecture explores the automated design of quantum circuits, where an algorithm searches for the optimal circuit structure (ansatz) that is both expressive for the task and resilient to hardware noise. QAS integrates the noise model into the objective function to find a trade-off between model capacity and noise-induced error, ultimately resulting in more efficient and robust QML models for NISQ devices.

## References

- EfficientML.ai Course | 2023 Fall | MIT 6.5940: [ Complete course video series ](https://youtube.com/playlist?list=PL80kAHvQbh-pT4lCkDT53zT8DKmhE0idB&si=Uu00N0zKopEixhw3).
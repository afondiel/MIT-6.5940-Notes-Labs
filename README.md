# MIT 6.5940 - Lecture Notes and Labs⚡️

A comprehensive course on **Efficient AI Computing**, covering state-of-the-art techniques for model compression, hardware-aware AI, and the efficient deployment of large generative models on resource-constrained devices.

## 🌟 Course Overview

The monumental scale of modern deep learning, especially Large Language Models (LLMs) and Diffusion Models, demands massive computational and memory resources. This course introduces the **system-algorithm co-design paradigm** necessary to enable powerful, yet accessible, AI applications on devices ranging from massive cloud TPUs to ultra-low-power microcontrollers (TinyML) and nascent Quantum Computers.

The focus is on achieving **maximal performance with minimal resource consumption**.

## 🎯 Key Learning Objectives

Upon completion of this course, you will be able to:

* **Shrink and Accelerate Models:** Master techniques like **Pruning, Quantization (INT8/INT4)**, and **Knowledge Distillation** to dramatically reduce model size and inference latency.
* **Design Efficient Architectures:** Utilize **Neural Architecture Search (NAS)**, specifically **Once-for-All (OFA)**, to automatically design hardware-aware networks.
* **Master LLM Efficiency:** Apply **Parameter-Efficient Fine-Tuning (PEFT)** methods like **LoRA** and **QLoRA** for efficient adaptation of multi-billion parameter models.
* **Optimize Distributed Systems:** Implement **Data, Pipeline, and Tensor Parallelism** for efficient training of models that exceed single-GPU memory.
* **Deploy to the Edge (TinyML):** Design models and system software (**MCUNet, TinyEngine**) capable of running complex AI on microcontrollers with Kilobytes of RAM.
* **Explore Future Computing:** Understand the fundamentals of **Quantum Machine Learning (QML)** and implement **Noise Mitigation** techniques for current NISQ hardware.

## 📚 Course Structure & Modules

The course is divided into four main modules:

### **Module 1: Foundational Efficiency (Lectures 1-6)**
| Lecture | Topic | Key Techniques Covered |
| :--- | :--- | :--- |
| **L1-L2** | Introduction & Neural Network Basics | FLOPs, MACs, Model Analysis, Performance Metrics. |
| **L3-L5** | **Pruning & Sparsity** | Fine-Grained/Unstructured, Structured Pruning, Rigging the Lottery (SET, ERK). |
| **L6** | **Knowledge Distillation** | Teacher-Student Paradigm, Distilling LLMs. |

### **Module 2: Advanced Model Compression & Design (Lectures 7-11)**
| Lecture | Topic | Key Techniques Covered |
| :--- | :--- | :--- |
| **L7-L9** | **Quantization** | Post-Training Quantization (PTQ), Quantization-Aware Training (QAT), INT4/Binary Networks. |
| **L10-L11** | **Neural Architecture Search (NAS)** | Differentiable NAS, One-Shot NAS, **Once-for-All (OFA)**. |

### **Module 3: Large Model and Distributed Systems (Lectures 12-20)**
| Lecture | Topic | Key Techniques Covered |
| :--- | :--- | :--- |
| **L12-L14** | **LLM Architectures & Quantization** | Transformers, GPT, LLM Post-Training Optimization (GPTQ). |
| **L15-L17** | **Parameter-Efficient Fine-Tuning (PEFT)** | **LoRA, QLoRA, Prompt/Prefix Tuning**. |
| **L18** | Application: Efficient Diffusion Models | Fast sampling, distillation for generative AI. |
| **L19-L20** | **Distributed Training** | Data Parallelism, **Zero Redundancy Optimizer (ZeRO)**, Pipeline & Tensor Parallelism. |

### **Module 4: Edge AI and Future Computing (Lectures 21-23)**
| Lecture | Topic | Key Techniques Covered |
| :--- | :--- | :--- |
| **L21** | **Basics of Quantum Computing** | Qubits, Superposition, Entanglement, Quantum Gates. |
| **L22** | **Quantum Machine Learning (QML)** | Variational Quantum Algorithms (VQAs), Quantum Kernels, Barren Plateaus. |
| **L23** | **Noise Robust Quantum ML** | **Zero Noise Extrapolation (ZNE)**, Measurement Error Mitigation (MEM). |

## ⚙️ Practical Lab Environment

All lab exercises are designed to provide **hands-on experience** with real-world frameworks.

* **LLM Deployment:** Hands-on experience deploying and running **QLoRA-tuned LLMs** (e.g., Llama-2) directly on a local GPU or CPU.
* **TinyML:** Utilizing the **TinyEngine** and **TensorFlow Lite Micro** frameworks for model deployment on simulated microcontroller environments.
* **QML:** Using **Qiskit** and **Pennylane** to build, train, and mitigate noise in variational quantum circuits.

## 💻 Tech Stack & Prerequisites

* **Programming:** Strong proficiency in **Python 3**.
* **Frameworks:** Experience with **PyTorch** (primary framework) or **TensorFlow**.
* **Math:** Comfort with **Linear Algebra, Calculus, and Probability**.
* **Prerequisites:** Familiarity with standard deep learning concepts (CNNs, RNNs, basic optimizers).

---

## References

- To see the original introductory lecture that sets the stage for the entire course, you can watch [EfficientML.ai Lecture 1 - Introduction (MIT 6.5940, Fall 2023)](https://www.youtube.com/watch?v=rCFvPEQTxKI).
http://googleusercontent.com/youtube_content/2
- Final project list (2023- 2024): [EfficientML.ai Project Ideas](https://docs.google.com/document/d/1QiCkCUr_1DnLNUCXUM3g0SQRIbVG5XyrQjdfsUPrIeA/edit?usp=sharing)



## Acknowledgements
- Firstly, to Professor [Song Han](https://hanlab.mit.edu/songhan) for his tremendous effort and passion on proving accessible for everyone 
- Big thanks to [Yifan Lu](https://github.com/yifanlu0227) for his effort on making [All Homeworks Labs Accessible](https://github.com/yifanlu0227/MIT-6.5940?tab=readme-ov-file) 
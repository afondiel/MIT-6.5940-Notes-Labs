# MIT 6.5940 - Notes and Labs⚡️

Notes and practical notebooks from [MIT 6.5940, (Fall 2023) : TinyML and Efficient Deep Learning Computing](https://www.youtube.com/watch?v=rCFvPEQTxKI) lectures.

## 🌟 Course Overview

This course introduces efficient deep learning computing techniques that enable powerful deep learning applications on resource-constrained devices. The main focus is on achieving **maximal performance with minimal resource consumption**.

## 🎯 Key Learning Objectives

Upon completion of this course, you will be able to:

* **Shrink and Accelerate Models:** Master techniques like **Pruning, Quantization (INT8/INT4)**, and **Knowledge Distillation** to dramatically reduce model size and inference latency.
* **Design Efficient Architectures:** Utilize **Neural Architecture Search (NAS)**, specifically **Once-for-All (OFA)**, to automatically design hardware-aware networks.
* **Master LLM Efficiency:** Apply **Parameter-Efficient Fine-Tuning (PEFT)** methods like **LoRA** and **QLoRA** for efficient adaptation of multi-billion parameter models.
* **Optimize Distributed Systems:** Implement **Data, Pipeline, and Tensor Parallelism** for efficient training of models that exceed single-GPU memory.
* **Deploy to the Edge (TinyML):** Design models and system software (**MCUNet, TinyEngine**) capable of running complex AI on microcontrollers with Kilobytes of RAM.
* **Explore Future Computing:** Understand the fundamentals of **Quantum Machine Learning (QML)** and implement **Noise Mitigation** techniques for current NISQ hardware.

## 💻 Tech Stack & Prerequisites

* **Programming:** Strong proficiency in **Python 3**.
* **Frameworks:** Experience with **PyTorch** (primary framework) or **TensorFlow**.
* **Math:** Comfort with **Linear Algebra, Calculus, and Probability**.
* **Prerequisites:** Familiarity with standard deep learning concepts (CNNs, RNNs, basic optimizers).

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

## 💻 Hands-on Lab Environment

All lab exercises are designed to provide **hands-on experience** with real-world frameworks:

* **LLM Deployment:** Hands-on experience deploying and running **QLoRA-tuned LLMs** (e.g., Llama-2) directly on a local GPU or CPU.
* **TinyML:** Utilizing the **TinyEngine** and **TensorFlow Lite Micro** frameworks for model deployment on simulated microcontroller environments.
* **QML:** Using **Qiskit** and **Pennylane** to build, train, and mitigate noise in variational quantum circuits.

| Lab | Key Concepts | Notebook | References |
| :--- | :--- | :--- | :--- |
| **The Baseline** | - **Parameter Counting** <br> - **FLOPs/MACs Calculation** <br> - **Unoptimized Latency Measurement** | **[L02\_NN\_Basics.ipynb](./)** | (Still to come)|
| **Model Pruning**| - **Unstructured Pruning** <br> - **Structured Pruning** <br> - Sparsity-Accuracy Trade-offs. | **[L03\_L04\_Pruning.ipynb](lab/notebooks/Lab1.ipynb)**| [Yifan Lu](https://github.com/yifanlu0227/MIT-6.5940)|
| **Quantization (PTQ)** | - **Post-Training Quantization (INT8)** <br> - Inference Speed-up <br> - Accuracy Degradation. | **[L08\_Quantization\_PTQ.ipynb](/lab/notebooks/Lab2.ipynb)** |[Yifan Lu](https://github.com/yifanlu0227/MIT-6.5940)|
| **Quantization (QAT)** | - **Quantization-Aware Training** for near-lossless INT8 accuracy. | **[L09\_Quantization\_QAT.ipynb](lab/notebooks/Lab4.ipynb)** | [Yifan Lu](https://github.com/yifanlu0227/MIT-6.5940) |
| **Neural Architecture Search (NAS)** | - **Differentiable NAS**, One-Shot NAS, **Once-for-All (OFA)** | **[L10\_L11\_NAS.ipynb](lab/notebooks/Lab3.ipynb)** |[Yifan Lu](https://github.com/yifanlu0227/MIT-6.5940)|
| **LLM Efficiency** | - **QLoRA** (4-bit + LoRA) for memory-efficient multi-billion parameter model fine-tuning. | **[L16\_LLM\_QLoRA\_Finetuning.ipynb](#)** |(Still to come)|
| **Edge AI** | - Model conversion to **TensorFlow Lite Micro** <br> - Memory/latency profiling on simulated MCUs. |**[L21\_TinyML\_Deployment.ipynb](#)** |(Still to come)|
| **Quantum ML** | - Implementing **Zero Noise Extrapolation (ZNE)** using Qiskit/Pennylane to combat hardware noise. |**[L23\_QML\_Noise\_Mitigation.ipynb](#)**|(Still to come)|


## ✨ Advanced Project Ideas (MIT 6.5940 Final Projects)

These projects represent state-of-the-art research challenges in efficient ML and are suitable for a student team to explore.


### 1. Project: TSM for Efficient Video Understanding (Temporal Shift Module)
* [cite_start]**Goal:** Address the challenge of efficient video analysis by leveraging **Temporal Shift Module (TSM)**, which captures temporal relationships without adding computational cost[cite: 8, 10].
* [cite_start]**Description:** TSM works by shifting part of the channels along the temporal dimension, facilitating information exchange among neighboring frames[cite: 9]. [cite_start]Projects could involve changing the backbone (e.g., from MobileNetV2) or applying TSM to a new video task like fall detection[cite: 14, 15].

### 2. Project: SIGE - Sparse Engine for Generative AI
* [cite_start]**Goal:** Accelerate image editing in deep generative models by avoiding the re-synthesis of unedited regions[cite: 34, 35].
* [cite_start]**Description:** **SIGE** (Sparse Inference GEnerator) is a sparse engine that caches and reuses feature maps from the original image to generate *only* the edited regions[cite: 36]. [cite_start]The project focuses on integrating SIGE with **Stable Diffusion XL (SDXL)** to assess and potentially achieve more significant speed improvements[cite: 37, 39].

### 3. Project: QServe for Online Quantized LLM Serving
* [cite_start]**Goal:** Achieve high-throughput, real-time serving of low-precision quantized LLMs (like INT4) in cloud-based settings[cite: 165, 182].
* **Description:** The project centers on implementing an **online, real-time serving system** using the **QServe** library, which utilizes the **QoQ (W4A8KV4)** quantization algorithm. [cite_start]The final objective is to build an online Gradio demo to serve these highly-efficient, quantized LLMs[cite: 168, 183, 185].

***

> **Full documentation and project details can be found [here](https://docs.google.com/document/d/1QiCkCUr_1DnLNUCXUM3g0SQRIbVG5XyrQjdfsUPrIeA/edit?usp=sharing)**.

## References

- The Youtube lecture series: [EfficientML.ai Lecture 1 - Introduction (MIT 6.5940, Fall 2023)](https://youtube.com/playlist?list=PL80kAHvQbh-pT4lCkDT53zT8DKmhE0idB&si=axVQj3jL6Ix1eyk6).
- Final project list (2023- 2024): [EfficientML.ai Project Ideas](https://docs.google.com/document/d/1QiCkCUr_1DnLNUCXUM3g0SQRIbVG5XyrQjdfsUPrIeA/edit?usp=sharing)

## 🙏 Acknowledgements

Special thanks to:

* **Professor [Song Han](https://github.com/songhan)** (MIT/HAN Lab) for his tremendous effort and passion in developing the **EfficientML.ai** framework and for making this cutting-edge research accessible to everyone.
* **[Yifan Lu](https://github.com/yifanlu0227)** for his dedication to making the course homework and lab materials publicly accessible and available for the community ([All Homeworks Labs Accessible](https://github.com/yifanlu0227/MIT-6.5940?tab=readme-ov-file)).


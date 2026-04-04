# MIT 6.5940 - Notes and Labs⚡️

Notes and hands-on labs from [MIT 6.5940, (Fall 2023) : TinyML and Efficient Deep Learning Computing](https://www.youtube.com/watch?v=rCFvPEQTxKI) lecture.

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

## 📚 Course Outiline (Updated from [Fall 2024](https://hanlab.mit.edu/courses/2024-fall-65940))

### Chapter 0: Introduction

| Lecture | Topic (notes) | Slide | Notebook | Reference |
| :--- | :--- | :--- | :--- | :--- |
| **L1** | [Introduction](./chapters/notes/L01\_Introduction.md) | [Slides](https://drive.google.com/drive/folders/1A3P6IBuS8wCzLlpdRiQBO9b1uoK3pnPf?usp=drive_link) | **—** | [Video](https://youtu.be/6cAmS-\_vEh8?si=2HtY0XwJEfj2gEGK) |
| **L2** | [Basics of Deep Learning](./chapters/notes/L02\_Basics\_of\_NN.md) | [Slides](https://drive.google.com/drive/folders/1A3P6IBuS8wCzLlpdRiQBO9b1uoK3pnPf?usp=drive_link) | **[L02\_NN\_Basics.ipynb](./lab/notebooks/Lab0.ipynb)** | [Video](https://youtu.be/Q9bdjoVx\_m4) |

### Chapter I: Efficient Inference

| Lecture | Topic (notes) | Slide | Notebook | Reference |
| :--- | :--- | :--- | :--- | :--- |
| **L3** | [Pruning and Sparsity (Part I)](./chapters/notes/L03\_Pruning\_I.md) | [Slides](https://drive.google.com/drive/folders/1A3P6IBuS8wCzLlpdRiQBO9b1uoK3pnPf?usp=drive_link) | **—** | [Video](#) |
| **L4** | [Pruning and Sparsity (Part II)](./chapters/notes/L04\_Pruning\_II.md) | [Slides](https://drive.google.com/drive/folders/1A3P6IBuS8wCzLlpdRiQBO9b1uoK3pnPf?usp=drive_link) | **[L03\_L04\_Pruning.ipynb](./lab/notebooks/Lab1.ipynb)** | [Video](#) |
| **L5** | [Quantization (Part I)](./chapters/notes/L05\_Quantization\_I.md) | [Slides](https://drive.google.com/drive/folders/1A3P6IBuS8wCzLlpdRiQBO9b1uoK3pnPf?usp=drive_link)| **—** | [Video](#) |
| **L6** | [Quantization (Part II)](./chapters/notes/L06\_Quantization\_II.md) | [Slides](https://drive.google.com/drive/folders/1A3P6IBuS8wCzLlpdRiQBO9b1uoK3pnPf?usp=drive_link) | **[L08\_Quantization\_PTQ.ipynb](./lab/notebooks/Lab2.ipynb)** | [Video](#) |
| **L7** | [Neural Architecture Search (Part I)](./chapters/notes/L07\_NAS\_I.md) | [Slides](https://drive.google.com/drive/folders/1A3P6IBuS8wCzLlpdRiQBO9b1uoK3pnPf?usp=drive_link) | **[Lab3\_NAS.ipynb](./lab/notebooks/Lab3.ipynb)** | [Video](#) |
| **L8** | [Neural Architecture Search (Part II)](./chapters/notes/L08\_NAS\_II.md) | [Slides](https://drive.google.com/drive/folders/1A3P6IBuS8wCzLlpdRiQBO9b1uoK3pnPf?usp=drive_link) | **—** | [Video](#) |
| **L9** | [Knowledge Distillation](./chapters/notes/L09\_Knowledge\_Distillation.md) | [Slides](https://drive.google.com/drive/folders/1A3P6IBuS8wCzLlpdRiQBO9b1uoK3pnPf?usp=drive_link) | **[L09\_Quantization\_QAT.ipynb](./labs/L09\_Quantization\_QAT.ipynb)** | [Video](#) |
| **L10** | [MCUNet: TinyML on Microcontrollers](./chapters/notes/L10\_MCUNet.md) | [Slides](https://drive.google.com/drive/folders/1A3P6IBuS8wCzLlpdRiQBO9b1uoK3pnPf?usp=drive_link) | **[L10\_L11\_NAS.ipynb](./labs/L10\_L11\_NAS.ipynb)** | [Video](#) |
| **L11** | [TinyEngine and Parallel Processing](./chapters/notes/L11\_TinyEngine.md) | [Slides](https://drive.google.com/drive/folders/1A3P6IBuS8wCzLlpdRiQBO9b1uoK3pnPf?usp=drive_link)| **[L21\_TinyML\_Deployment.ipynb](./labs/L21\_TinyML\_Deployment.ipynb)** | [Video](#) |

### Chapter II: Domain-Specific Optimization

| Lecture | Topic (notes) | Slide | Notebook | Reference |
| :--- | :--- | :--- | :--- | :--- |
| **L12** | [Transformer and LLM](./chapters/notes/L12\_Transformer\_LLM.md)       | [Slides](https://drive.google.com/drive/folders/1A3P6IBuS8wCzLlpdRiQBO9b1uoK3pnPf?usp=drive_link) | **—** | [Video](#) |
| **L13** | [Efficient LLM Deployment](./chapters/notes/L13\_LLM\_Deployment.md)   | [Slides](https://drive.google.com/drive/folders/1A3P6IBuS8wCzLlpdRiQBO9b1uoK3pnPf?usp=drive_link) |**[Lab4\_LLM_Quantization.ipynb](./lab/notebooks/Lab4.ipynb)**| [Video](#) |
| **L14** | [LLM Post Training](./chapters/notes/L14\_LLM\_Post\_Training.md)      | [Slides](https://drive.google.com/drive/folders/1A3P6IBuS8wCzLlpdRiQBO9b1uoK3pnPf?usp=drive_link) | **—** | [Video](#) |
| **L15** | [Long Context LLM](./chapters/notes/L15\_Long\_Context\_LLM.md)        | [Slides](https://drive.google.com/drive/folders/1A3P6IBuS8wCzLlpdRiQBO9b1uoK3pnPf?usp=drive_link) | **—** | [Video](#) |
| **L16** | [Vision Transformer](./chapters/notes/L16\_Vision\_Transformer.md)     | [Slides](https://drive.google.com/drive/folders/1A3P6IBuS8wCzLlpdRiQBO9b1uoK3pnPf?usp=drive_link) | **[L16\_LLM\_QLoRA\_Finetuning.ipynb](./labs/L16\_LLM\_QLoRA\_Finetuning.ipynb)** | [Video](#) |
| **L17** | [GAN, Video, and Point Cloud](./chapters/notes/L17\_GAN\_Video\_3D.md) | [Slides](https://drive.google.com/drive/folders/1A3P6IBuS8wCzLlpdRiQBO9b1uoK3pnPf?usp=drive_link) | **—** | [Video](#) |
| **L18** | [Diffusion Model](./chapters/notes/L18\_Diffusion\_Model.md)           | [Slides](https://drive.google.com/drive/folders/1A3P6IBuS8wCzLlpdRiQBO9b1uoK3pnPf?usp=drive_link) | **—** | [Video](#) |
| **LA1** *| [Audio Transformers and Efficient Speech Recognition](./chapters/notes/LA1\_Audio\_Transformers\_ASR.md) | - | - | wav2vec 2.0, Whisper, Conformer |
| **LA2** *| [Efficient Speech Synthesis and Audio Generation](./chapters/notes/LA2\_Speech\_Synthesis\_Audio\_Generation.md) | - | - | WaveNet, FastSpeech, VALL-E, AudioLDM |
| **LA3** *| [Audio-Language Models and Sound Understanding](./chapters/notes/LA3\_Audio\_Language\_Models.md) | - | - | CLAP, SALMONN, Qwen-Audio |

### Chapter III: Efficient Training

| Lecture | Topic (notes) | Slide | Notebook | Reference |
| :--- | :--- | :--- | :--- | :--- |
| **L19** | [Distributed Training (Part I)](./chapters/notes/L19\_Distributed\_I.md)                  | [Slides](https://drive.google.com/drive/folders/1A3P6IBuS8wCzLlpdRiQBO9b1uoK3pnPf?usp=drive_link) | **[Lab5\_LLM_at_Edge.ipynb](./lab/notebooks/lab5-url.md)** | [Video](#) |
| **L20** | [Distributed Training (Part II)](./chapters/notes/L20\_Distributed\_II.md)                | [Slides](https://drive.google.com/drive/folders/1A3P6IBuS8wCzLlpdRiQBO9b1uoK3pnPf?usp=drive_link) | **—** | [Video](#) |
| **L21** | [On-Device Training and Transfer Learning](./chapters/notes/L21\_On\_Device\_Training.md) | [Slides](https://drive.google.com/drive/folders/1A3P6IBuS8wCzLlpdRiQBO9b1uoK3pnPf?usp=drive_link) | **—** | [Video](#) |


### Chapter IV: Advanced Topics

| Lecture | Topic (notes) | Slide | Notebook | Reference |
| :--- | :--- | :--- | :--- | :--- |
| **L22** | [Course Summary + Quantum ML I](./chapters/notes/L22\_Summary\_QML\_I.md) | [Slides](https://drive.google.com/drive/folders/1A3P6IBuS8wCzLlpdRiQBO9b1uoK3pnPf?usp=drive_link) | **—** | [Video](#) |
| **L23** | [Quantum Machine Learning II](./chapters/notes/L23\_QML\_II.md)           | [Slides](https://drive.google.com/drive/folders/1A3P6IBuS8wCzLlpdRiQBO9b1uoK3pnPf?usp=drive_link) | **[L23\_QML\_Noise\_Mitigation.ipynb](./labs/L23\_QML\_Noise\_Mitigation.ipynb)** | [Video](#) |
| **L24** | [Final Project Presentation](./chapters/notes/L24\_Project\_Pres.md)      | [Slides](https://drive.google.com/drive/folders/1A3P6IBuS8wCzLlpdRiQBO9b1uoK3pnPf?usp=drive_link) | **—** | [Video](#) |
| **L25** | [Final Project Presentation](./chapters/notes/L25\_Project\_Pres.md)      | [Slides](https://drive.google.com/drive/folders/1A3P6IBuS8wCzLlpdRiQBO9b1uoK3pnPf?usp=drive_link) | **—** | [Video](#) |
| **L26** | [Final Project Presentation](./chapters/notes/L26\_Project\_Pres.md)      | [Slides](https://drive.google.com/drive/folders/1A3P6IBuS8wCzLlpdRiQBO9b1uoK3pnPf?usp=drive_link) | **—** | [Video](#) |

## 🔊 LA1–LA3: Community Research Extension for Audio Domain

> [!NOTE]  
> **Note:** The lectures prefixed **LA** [(LA1–LA3)](chapters/notes/LA1_Audio_Transformers_ASR.md) are **not** part of the official MIT 6.5940 curriculum. They are community-designed research notes that I created to extend the course's domain-specific coverage to the **audio modality** — an area largely absent from the original syllabus despite its critical importance for Edge AI.

### Motivation

The official course covers efficient techniques for **Language** (L12–L15) and **Vision** (L16–L18), but the **audio/sound** modality was left out. As an [Edge AI Engineer](https://afondiel.github.io/), I found this gap significant for several reasons:

1. **Audio is the dominant edge modality.** Billions of devices (earbuds, smart speakers, hearing aids, vehicles, MCUs) rely on real-time audio processing — often under tighter latency, memory, and power constraints than vision or text.
2. **The same efficiency playbook applies.** Every core technique taught in Chapters I–III (pruning, quantization, NAS, knowledge distillation, distributed training) transfers directly to audio models — yet the domain has unique challenges (**long sequences, streaming requirements, sample-rate bottlenecks**) that deserve dedicated coverage.
3. **Rapid convergence with LLMs.** Audio-Language Models (ALMs) are following the exact same trajectory as Vision-Language Models (VLMs): `frozen encoder → bridge module → LLM backbone`. Understanding this pattern across all three modalities gives a complete picture of efficient multimodal AI at the edge.

### Structure

These notes follow **Professor [Song Han](https://github.com/songhan)'s pedagogical framework**: start from the breakthrough model that defined the domain, scale it up, then systematically compress it back down for deployment using the course's core techniques.

| Audio Lecture | Parallel to | Breakthrough Paper |
|---|---|---|
| **LA1** — Audio Transformers & ASR | L12–L13 (Transformer → LLM Deployment) | **wav2vec 2.0** (*Baevski et al., 2020*) |
| **LA2** — Speech Synthesis & Audio Generation | L17–L18 (GAN → Diffusion) | **WaveNet** (*van den Oord et al., 2016*) |
| **LA3** — Audio-Language Models & Sound Understanding | L12 Multimodal + L16 (ViT) | **CLAP** (*Elizalde et al., 2023*) |

## 💻 Hands-on Labs & Advanced Project Ideas (MIT 6.5940 Final Projects)

All lab exercises are designed to provide **hands-on experience** with real-world frameworks:

- **LLM Deployment:** Hands-on experience deploying and running **QLoRA-tuned LLMs** (e.g., Llama-2) directly on a local GPU or CPU.
- **TinyML:** Utilizing the **TinyEngine** and **TensorFlow Lite Micro** frameworks for model deployment on simulated microcontroller environments.
- **QML:** Using **Qiskit** and **Pennylane** to build, train, and mitigate noise in variational quantum circuits.

For further advanced projects the course provided a set of state-of-the-art research challenges in efficient ML to explore.


### 1. Project: TSM for Efficient Video Understanding (Temporal Shift Module)
* **Goal:** Address the challenge of efficient video analysis by leveraging **Temporal Shift Module (TSM)**, which captures temporal relationships without adding computational cost[cite: 8, 10].
* **Description:** TSM works by shifting part of the channels along the temporal dimension, facilitating information exchange among neighboring frames[cite: 9]. Projects could involve changing the backbone (e.g., from MobileNetV2) or applying TSM to a new video task like fall detection[cite: 14, 15].

### 2. Project: SIGE - Sparse Engine for Generative AI
* **Goal:** Accelerate image editing in deep generative models by avoiding the re-synthesis of unedited regions[cite: 34, 35].
* **Description:** **SIGE** (Sparse Inference GEnerator) is a sparse engine that caches and reuses feature maps from the original image to generate *only* the edited regions[cite: 36]. The project focuses on integrating SIGE with **Stable Diffusion XL (SDXL)** to assess and potentially achieve more significant speed improvements[cite: 37, 39].

### 3. Project: QServe for Online Quantized LLM Serving
* **Goal:** Achieve high-throughput, real-time serving of low-precision quantized LLMs (like INT4) in cloud-based settings[cite: 165, 182].
* **Description:** The project centers on implementing an **online, real-time serving system** using the **QServe** library, which utilizes the **QoQ (W4A8KV4)** quantization algorithm. The final objective is to build an online Gradio demo to serve these highly-efficient, quantized LLMs[cite: 168, 183, 185].

> **Full documentation and project details can be found [here](https://docs.google.com/document/d/1QiCkCUr_1DnLNUCXUM3g0SQRIbVG5XyrQjdfsUPrIeA/edit?usp=sharing)**.


## ⚠️ Disclaimer

> [!IMPORTANT]
> The Audio extension notes (LA1–LA3) are **not** official course material — they are community research notes that I designed by applying Professor [Han](https://hanlab.mit.edu/songhan)'s pedagogical framework and the course's core efficiency principles to the audio domain. All credit for the teaching methodology, structure, and foundational techniques goes to him and the [MIT HAN Lab team](https://hanlab.mit.edu/team).

## 🙏 Acknowledgements

Special thanks to: 

- **Professor [Song Han](https://github.com/songhan)** (MIT/HAN Lab) for his tremendous effort and passion in developing the **[EfficientML.ai](https://hanlab.mit.edu/)** framework and ecosystem, and for making them accessible to everyone. 
- **[Yifan Lu](https://github.com/yifanlu0227)** for sharing [all homework labs](https://github.com/yifanlu0227/MIT-6.5940?tab=readme-ov-file).


## References

- Course Youtube Playlist: [EfficientML.ai Course | 2023 Fall | MIT 6.5940](https://youtube.com/playlist?list=PL80kAHvQbh-pT4lCkDT53zT8DKmhE0idB&si=Uu00N0zKopEixhw3)
- Course Youtube Playlist (NEW): [EfficientML.ai Course | 2024 Fall | MIT 6.5940](https://youtube.com/playlist?list=PL80kAHvQbh-qGtNc54A6KW4i4bkTPjiRF&si=7ttsXhRJzkk_H7BD)
- Course Prerequisites: [pdf](https://drive.google.com/drive/folders/15vk9PbG5EHK5u4mML7InOAQbyuvmxnZl?usp=sharing)
- All slides available: [here](https://drive.google.com/drive/folders/1A3P6IBuS8wCzLlpdRiQBO9b1uoK3pnPf?usp=drive_link)
- Final project list (2023- 2024): [EfficientML.ai Project Ideas (Fall 2024)](https://docs.google.com/document/d/1QiCkCUr_1DnLNUCXUM3g0SQRIbVG5XyrQjdfsUPrIeA/edit?usp=sharing)

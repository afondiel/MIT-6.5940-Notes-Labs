# L5 Paper Summary

This set of references focuses on **Quantization**, ranging from foundational surveys to extreme binarization and ternarization techniques used in TinyML.

### **Foundational Surveys & Energy Theory**

* **[1] Model Compression and Hardware Acceleration for Neural Networks: A Comprehensive Survey** [Deng et al., IEEE 2020]
**[Direct PDF (ArXiv)](https://www.google.com/search?q=https://arxiv.org/pdf/1810.06601.pdf)** | [IEEE Xplore](https://ieeexplore.ieee.org/document/9043731)
* **[2] Computing's Energy Problem (and What We Can Do About it)** [Horowitz, M., IEEE ISSCC 2014]
**[Direct PDF](https://www.google.com/search?q=https://ieeexplore.ieee.org/stamp/stamp.jsp%3Ftp%3D%26arnumber%3D6757323)** | [ResearchGate Link](https://www.researchgate.net/publication/271463146_11_Computing's_energy_problem_and_what_we_can_do_about_it)
* **[4] Neural Network Distiller: Quantization Algorithms** [Intel AI Lab]
**[Documentation Page](https://intellabs.github.io/distiller/algo_quantization.html)** — *A practical guide to implementing these papers.*

---

### **Core Quantization Frameworks**

* **[3] Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding** [Han et al., ICLR 2016]
**[Direct PDF (ArXiv)](https://www.google.com/search?q=https://arxiv.org/pdf/1510.00149.pdf)**
* **[5] Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference** [Jacob et al., CVPR 2018]
**[Direct PDF (CVF)](https://openaccess.thecvf.com/content_cvpr_2018/papers/Jacob_Quantization_and_Training_CVPR_2018_paper.pdf)** — *This is the foundational paper for TensorFlow Lite's 8-bit quantization.*

---

### **Binary & Ternary Networks (Extreme Compression)**

* **[6] BinaryConnect: Training Deep Neural Networks with Binary Weights** [Courbariaux et al., NeurIPS 2015]
**[Direct PDF (ArXiv)](https://www.google.com/search?q=https://arxiv.org/pdf/1511.00363.pdf)**
* **[7] Binarized Neural Networks: Training with Weights and Activations Constrained to +1 or -1** [Courbariaux et al., ArXiv 2016]
**[Direct PDF (ArXiv)](https://www.google.com/search?q=https://arxiv.org/pdf/1602.02830.pdf)**
* **[8] XNOR-Net: ImageNet Classification using Binary Convolutional Neural Networks** [Rastegari et al., ECCV 2016]
**[Direct PDF (ArXiv)](https://www.google.com/search?q=https://arxiv.org/pdf/1603.05279.pdf)**
* **[9] Ternary Weight Networks** [Li et al., ArXiv 2016]
**[Direct PDF (ArXiv)](https://www.google.com/search?q=https://arxiv.org/pdf/1605.04711.pdf)**
* **[10] Trained Ternary Quantization** [Zhu et al., ICLR 2017]
**[Direct PDF (ArXiv)](https://www.google.com/search?q=https://arxiv.org/pdf/1612.01064.pdf)**

### **Key takeaway for your studies:**

The transition from **Paper [5]** (8-bit integer) to **Papers [6-10]** (1-bit or 2-bit) represents the jump from "industry standard" efficiency to "experimental" edge research. While 8-bit is widely supported on mobile CPUs/GPUs, binary and ternary networks often require custom FPGA or ASIC hardware to see real-world speedups.

# L4 Paper Summary

This list of papers focused on **Pruning Frameworks, AutoML, and Sparse Hardware Accelerators.**

### **Pruning & Sparsity Theory**

* **[1] Learning Both Weights and Connections for Efficient Neural Network** [Han et al., NeurIPS 2015]
**[Direct PDF](https://www.google.com/search?q=https://proceedings.neurips.cc/paper/2015/file/ae0eb3eed39d2bcef4622b2499a05fe6-Paper.pdf)**
* **[2] Exploring the Granularity of Sparsity in Convolutional Neural Networks** [Mao et al., CVPR-W 2017]
**[Direct PDF](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w29/papers/Mao_Exploring_the_Granularity_CVPR_2017_paper.pdf)**
* **[3] Learning Structured Sparsity in Deep Neural Networks** [Wen et al., NeurIPS 2016]
**[Direct PDF](https://www.google.com/search?q=https://proceedings.neurips.cc/paper/2016/file/41bfd20a303edc30df2fd18305a8f3c1-Paper.pdf)**
* **[4] Learning Efficient Convolutional Networks through Network Slimming** [Liu et al., ICCV 2017]
**[Direct PDF (ArXiv)](https://www.google.com/search?q=https://arxiv.org/pdf/1708.06519.pdf)**
* **[5] A Systematic DNN Weight Pruning Framework using ADMM** [Zhang et al., ECCV 2018]
**[Direct PDF (ArXiv)](https://www.google.com/search?q=https://arxiv.org/pdf/1804.03294.pdf)**

---

### **AutoML & Optimization**

* **[6] AMC: AutoML for Model Compression and Acceleration on Mobile Devices** [He et al., ECCV 2018]
**[Direct PDF (ArXiv)](https://www.google.com/search?q=https://arxiv.org/pdf/1802.03494.pdf)**
* **[11] Accelerating Sparse Deep Neural Networks** [Mishra et al., arXiv 2021]
**[Direct PDF (ArXiv)](https://www.google.com/search?q=https://arxiv.org/pdf/2104.08378.pdf)**

---

### **Hardware Accelerators & Industry Standards**

* **[7] Accelerating Inference with Sparsity Using the NVIDIA Ampere Architecture** [NVIDIA Whitepaper]
**[Direct PDF](https://developer.download.nvidia.com/video/gputechconf/gtc/2020/presentations/s22085-accelerating-sparsity-in-the-nvidia-ampere-architecture%E2%80%8B.pdf)**
* **[8] EIE: Efficient Inference Engine on Compressed Deep Neural Network** [Han et al., ISCA 2016]
**[Direct PDF (ArXiv)](https://www.google.com/search?q=https://arxiv.org/pdf/1602.01527.pdf)**
* **[9] ESE: Efficient Speech Recognition Engine with Sparse LSTM on FPGA** [Han et al., FPGA 2017]
**[Direct PDF](https://dl.acm.org/doi/pdf/10.1145/3020078.3021745)**
* **[10] Block Sparse Format** [NVIDIA Documentation/Blog, 2021]
[View Official Documentation](https://www.google.com/search?q=https://docs.nvidia.com/cuda/cublas/index.html%23cublas-extension-api-block-sparse-matrix-multiplication) | [Related Blog on Structured Sparsity](https://developer.nvidia.com/blog/accelerating-inference-with-sparsity-using-ampere-and-tensorrt/)

---

### **Quick Summary of Hardware Sparsity**

These papers track the evolution of how we handle "empty" space in AI models:

1. **Fine-grained (Ref 1):** Flexible but hard for traditional CPUs/GPUs to speed up.
2. **Structured/Slimming (Ref 3, 4):** Removes whole rows or channels; very fast on standard hardware.
3. **NVIDIA 2:4 Sparsity (Ref 7, 10):** A middle ground where for every 4 weights, at least 2 must be zero, allowing Ampere GPUs to double throughput.


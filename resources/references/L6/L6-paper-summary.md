# L6 Paper Summary

This new list moves deeper into **Quantization-Aware Training (QAT)**, **Post-Training Quantization (PTQ)**, and the math behind training models with extremely low precision (1-4 bits).

### **I. Foundational 8-bit & Integer Quantization**

* **[1] Deep Compression** [Han et al., ICLR 2016]
**[Direct PDF (ArXiv)](https://www.google.com/search?q=https://arxiv.org/pdf/1510.00149.pdf)**
* **[2] Neural Network Distiller: Quantization Algorithms** [Intel AI Lab]
**[Documentation](https://intellabs.github.io/distiller/algo_quantization.html)**
* **[3] Quantization and Training for Integer-Arithmetic-Only Inference** [Jacob et al., CVPR 2018]
**[Direct PDF (CVF)](https://openaccess.thecvf.com/content_cvpr_2018/papers/Jacob_Quantization_and_Training_CVPR_2018_paper.pdf)**
* **[7] Quantizing Deep Convolutional Networks for Efficient Inference: A Whitepaper** [Krishnamoorthi, arXiv 2018]
**[Direct PDF (ArXiv)](https://www.google.com/search?q=https://arxiv.org/pdf/1806.08342.pdf)**

---

### **II. Post-Training Quantization (PTQ) & Industry Tools**

These methods are "rapid-deployment" because they don't require retraining the model from scratch.

* **[4] Data-Free Quantization (Weight Equalization & Bias Correction)** [Nagel et al., ICCV 2019]
**[Direct PDF (CVF)](https://openaccess.thecvf.com/content_ICCV_2019/papers/Nagel_Data-Free_Quantization_Through_Weight_Equalization_and_Bias_Correction_ICCV_2019_paper.pdf)**
* **[5] Post-Training 4-Bit Quantization of Convolution Networks** [Banner et al., NeurIPS 2019]
**[Direct PDF (NeurIPS)](https://www.google.com/search?q=https://proceedings.neurips.cc/paper/2019/file/c0a62e1338a02f371443d08595447f51-Paper.pdf)**
* **[6] 8-bit Inference with TensorRT** [Migacz, GTC 2017]
**[Presentation Slides (NVIDIA)](https://www.google.com/search?q=http://on-demand.gputechconf.com/gtc/2017/presentation/s7310-8-bit-inference-with-tensorrt.pdf)**

---

### **III. The Math of Training Low-Bit/Binary Networks**

Training models with discrete weights (like 0 or 1) is tricky because gradients are usually zero. These papers solve that with the **Straight-Through Estimator (STE)**.

* **[8] Neural Networks for Machine Learning** [Hinton, 2012]
**[YouTube Lecture Playlist](https://www.youtube.com/playlist?list=PLoRl3Ht4JOcdU872GhiYWf6jwrk_SNhz9)** — *The foundational Coursera course.*
* **[9] Estimating Gradients Through Stochastic Neurons** [Bengio, arXiv 2013]
**[Direct PDF (ArXiv)](https://www.google.com/search?q=https://arxiv.org/pdf/1308.3432.pdf)**
* **[10] Binarized Neural Networks (BNNs)** [Courbariaux et al., ArXiv 2016]
**[Direct PDF (ArXiv)](https://www.google.com/search?q=https://arxiv.org/pdf/1602.02830.pdf)**
* **[11] DoReFa-Net: Low Bitwidth Convolutional Neural Networks** [Zhou et al., arXiv 2016]
**[Direct PDF (ArXiv)](https://www.google.com/search?q=https://arxiv.org/pdf/1606.06160.pdf)**
* **[14] Towards Accurate Binary Convolutional Neural Network** [Lin et al., NeurIPS 2017]
**[Direct PDF (NeurIPS)](https://www.google.com/search?q=https://proceedings.neurips.cc/paper/2017/file/07096fb2d377b282460ed6e60b7952e4-Paper.pdf)**

---

### **IV. Advanced Quantization (Mixed-Precision & PACT)**

* **[12] PACT: Parameterized Clipping Activation for Quantized NNs** [Choi et al., arXiv 2018]
**[Direct PDF (ArXiv)](https://www.google.com/search?q=https://arxiv.org/pdf/1805.06085.pdf)**
* **[13] WRPN: Wide Reduced-Precision Networks** [Mishra et al., ICLR 2018]
**[Direct PDF (ArXiv)](https://www.google.com/search?q=https://arxiv.org/pdf/1709.01134.pdf)**
* **[15] Incremental Network Quantization (INQ)** [Zhou et al., ICLR 2017]
**[Direct PDF (ArXiv)](https://www.google.com/search?q=https://arxiv.org/pdf/1702.03044.pdf)**
* **[16] HAQ: Hardware-Aware Automated Quantization** [Wang et al., CVPR 2019]
**[Direct PDF (ArXiv)](https://www.google.com/search?q=https://arxiv.org/pdf/1811.08886.pdf)**

---

### **Quick Look: Why "Wide" Networks? (Ref 13)**

One interesting takeaway from **WRPN (Ref 13)** is that if you reduce weight precision to 1 or 2 bits, you can often recover the lost accuracy by simply making the layers "wider" (more channels). It's a trade-off: **Lower Precision + More Parameters = Same Accuracy but faster execution.**


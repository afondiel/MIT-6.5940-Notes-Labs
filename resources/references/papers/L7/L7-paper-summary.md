# L7 Paper Summary

This extensive list covers the evolution of modern Computer Vision, from the "Big Bang" of Deep Learning (AlexNet) to the cutting-edge **Neural Architecture Search (NAS)** and **IoT-specific models** (MCUNet).

Here are the links to the papers:

### **I. Manual Architecture Design (Classic & Mobile)**

These papers moved the field from "just make it deeper" to "make it efficient."

* **[1] Deep Residual Learning (ResNet)**
**[Direct PDF](https://www.google.com/search?q=https://arxiv.org/pdf/1512.03385.pdf)**
* **[2] AlexNet (The 2012 Breakthrough)**
**[Direct PDF](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)**
* **[3] VGG Networks**
**[Direct PDF](https://www.google.com/search?q=https://arxiv.org/pdf/1409.1556.pdf)**
* **[4] SqueezeNet (50x fewer parameters)**
**[Direct PDF](https://www.google.com/search?q=https://arxiv.org/pdf/1602.07360.pdf)**
* **[5] ResNeXt (Aggregated Transformations)**
**[Direct PDF](https://www.google.com/search?q=https://arxiv.org/pdf/1611.05431.pdf)**
* **[6] MobileNetV1 (Depthwise Separable Conv)**
**[Direct PDF](https://www.google.com/search?q=https://arxiv.org/pdf/1704.04861.pdf)**
* **[7] MobileNetV2 (Inverted Residuals)**
**[Direct PDF](https://www.google.com/search?q=https://arxiv.org/pdf/1801.04381.pdf)**
* **[8] ShuffleNet (Channel Shuffle)**
**[Direct PDF](https://www.google.com/search?q=https://arxiv.org/pdf/1707.01083.pdf)**
* **[15] Designing Network Design Spaces (RegNet)**
**[Direct PDF](https://www.google.com/search?q=https://arxiv.org/pdf/2003.13678.pdf)**
* **[22] EfficientNet (Compound Scaling)**
**[Direct PDF](https://www.google.com/search?q=https://arxiv.org/pdf/1905.11946.pdf)**

---

### **II. Neural Architecture Search (NAS)**

Instead of humans designing the model, these papers use AI to find the best architecture for a specific goal.

* **[9] NAS Survey**
**[Direct PDF](https://jmlr.org/papers/volume20/18-598/18-598.pdf)**
* **[10] NASNet (Transferable Architectures)**
**[Direct PDF](https://www.google.com/search?q=https://arxiv.org/pdf/1707.07012.pdf)**
* **[11] DARTS (Differentiable Search)**
**[Direct PDF](https://www.google.com/search?q=https://arxiv.org/pdf/1806.09055.pdf)**
* **[12] MnasNet (Platform-Aware)**
**[Direct PDF](https://www.google.com/search?q=https://arxiv.org/pdf/1807.11626.pdf)**
* **[13] ProxylessNAS (Direct Search on Hardware)**
**[Direct PDF](https://www.google.com/search?q=https://arxiv.org/pdf/1812.00332.pdf)**
* **[14] FBNet (Differentiable NAS for Mobile)**
**[Direct PDF](https://www.google.com/search?q=https://arxiv.org/pdf/1812.03443.pdf)**
* **[16] Single Path One-Shot NAS**
**[Direct PDF](https://www.google.com/search?q=https://arxiv.org/pdf/1904.00420.pdf)**
* **[23] NAS with Reinforcement Learning**
**[Direct PDF](https://www.google.com/search?q=https://arxiv.org/pdf/1611.01578.pdf)**
* **[25] AmoebaNet (Evolution-based)**
**[Direct PDF](https://www.google.com/search?q=https://arxiv.org/pdf/1802.01548.pdf)**

---

### **III. System-Level & Domain-Specific Efficiency**

Focuses on specialized deployment (3D, IoT, Object Detection).

* **[17] Once-for-All (OFA) (Train once, deploy anywhere)**
**[Direct PDF](https://www.google.com/search?q=https://arxiv.org/pdf/1908.09791.pdf)**
* **[18] Auto-DeepLab (Semantic Segmentation)**
**[Direct PDF](https://www.google.com/search?q=https://arxiv.org/pdf/1901.02985.pdf)**
* **[19] NAS-FPN (Object Detection)**
**[Direct PDF](https://www.google.com/search?q=https://arxiv.org/pdf/1904.07392.pdf)**
* **[20] Randomly Wired Neural Networks**
**[Direct PDF](https://www.google.com/search?q=https://arxiv.org/pdf/1904.01569.pdf)**
* **[21] MCUNet (Deep Learning on $kB$ of memory)**
**[Direct PDF](https://www.google.com/search?q=https://arxiv.org/pdf/2007.10319.pdf)**
* **[24] PVNAS (3D Point-Voxel Search)**
**[Direct PDF](https://www.google.com/search?q=https://arxiv.org/pdf/2008.03809.pdf)**

---

### **Quick Learning Guide**

If you are studying for **TinyML (MIT 6.5940)**:

1. **MobileNetV2 [7]** is the "industry standard" for mobile vision. Understand the **Inverted Residual**—it's likely on your exam.
2. **Once-for-All [17]** and **MCUNet [21]** represent the modern frontier where the model and the compiler are designed together.
3. **EfficientNet [22]** is your go-to when you have a bit more compute but want maximum accuracy for every "flop" spent.



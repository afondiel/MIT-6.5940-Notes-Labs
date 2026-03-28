# L9 Paper Summary

This latest set of references from **MIT 6.5940** covers two massive pillars of efficient ML: **Knowledge Distillation (KD)** (the "Teacher-Student" paradigm) and **Advanced Data Augmentation/Regularization**.

### **I. Knowledge Distillation (The Core Papers)**

These papers explore how a small "student" model can learn from a large, accurate "teacher" model.

* **[2] Distilling the Knowledge in a Neural Network** [Hinton et al., NeurIPS 2014]
**[Direct PDF (ArXiv)](https://www.google.com/search?q=https://arxiv.org/pdf/1503.02531.pdf)** — *The paper that defined the field.*
* **[3] Knowledge Distillation: A Survey** [Gou et al., IJCV 2021]
**[Direct PDF (ArXiv)](https://www.google.com/search?q=https://arxiv.org/pdf/2006.05525.pdf)**
* **[4] Do Deep Nets Really Need to be Deep?** [Ba and Caruana, NeurIPS 2014]
**[Direct PDF](https://www.google.com/search?q=https://proceedings.neurips.cc/paper/2014/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf)**
* **[5] FitNets: Hints for Thin Deep Nets** [Romero et al., ICLR 2015]
**[Direct PDF (ArXiv)](https://www.google.com/search?q=https://arxiv.org/pdf/1412.6550.pdf)** — *Introduces "intermediate" layer hints.*

---

### **II. Advanced KD Techniques (Attention, Features, Relations)**

Moving beyond simple output matching to matching internal "logic."

* **[6] Like What You Like (Neuron Selectivity Transfer)** [Huang et al., 2017]
**[Direct PDF (ArXiv)](https://www.google.com/search?q=https://arxiv.org/pdf/1707.01219.pdf)**
* **[7] Paying More Attention to Attention** [Zagoruyko et al., ICLR 2017]
**[Direct PDF (ArXiv)](https://www.google.com/search?q=https://arxiv.org/pdf/1612.03928.pdf)**
* **[8] Paraphrasing Complex Network (Factor Transfer)** [Kim et al., NeurIPS 2018]
**[Direct PDF (ArXiv)](https://www.google.com/search?q=https://arxiv.org/pdf/1802.04977.pdf)**
* **[10] A Gift from Knowledge Distillation (FSP Matrix)** [Yim et al., CVPR 2017]
**[Direct PDF (CVF)](https://openaccess.thecvf.com/content_cvpr_2017/papers/Yim_A_Gift_From_CVPR_2017_paper.pdf)**
* **[11] Relational Knowledge Distillation** [Park et al., CVPR 2019]
**[Direct PDF (ArXiv)](https://www.google.com/search?q=https://arxiv.org/pdf/1904.05068.pdf)**

---

### **III. Modern KD Variants (Self, Mutual, and Specialized)**

* **[13] Born-Again Neural Networks** [Furlanello et al., ICML 2018]
**[Direct PDF (ArXiv)](https://www.google.com/search?q=https://arxiv.org/pdf/1805.04770.pdf)**
* **[14] Deep Mutual Learning** [Zhang et al., CVPR 2018]
**[Direct PDF (ArXiv)](https://www.google.com/search?q=https://arxiv.org/pdf/1706.00384.pdf)**
* **[16] Be Your Own Teacher (Self-Distillation)** [Zhang et al., ICCV 2019]
**[Direct PDF (ArXiv)](https://www.google.com/search?q=https://arxiv.org/pdf/1905.08094.pdf)**
* **[21] MobileBERT** [Sun et al., ACL 2020]
**[Direct PDF (ArXiv)](https://www.google.com/search?q=https://arxiv.org/pdf/2004.02984.pdf)**

---

### **IV. Data Augmentation & Regularization**

How to prevent overfitting when working with the smaller models typical of TinyML.

* **[22] Improved Regularization with Cutout** [DeVries et al., 2017]
**[Direct PDF (ArXiv)](https://www.google.com/search?q=https://arxiv.org/pdf/1708.04552.pdf)**
* **[23] mixup: Beyond Empirical Risk Minimization** [Zhang et al., ICLR 2018]
**[Direct PDF (ArXiv)](https://www.google.com/search?q=https://arxiv.org/pdf/1710.09412.pdf)**
* **[24] AutoAugment: Learning Augmentation Policies from Data** [Cubuk et al., CVPR 2019]
**[Direct PDF (ArXiv)](https://www.google.com/search?q=https://arxiv.org/pdf/1805.09501.pdf)**
* **[26] DropBlock: A Regularization Method for ConvNets** [Ghiasi et al., NeurIPS 2018]
**[Direct PDF (ArXiv)](https://www.google.com/search?q=https://arxiv.org/pdf/1810.12890.pdf)**

---

### **V. Task-Specific Efficiency (Detection & Segmentation)**

* **[1] Network Augmentation for Tiny Deep Learning** [Cai et al., ICLR 2022]
**[Direct PDF (ArXiv)](https://www.google.com/search?q=https://arxiv.org/pdf/2110.08818.pdf)**
* **[17] Learning Efficient Detection Models with KD** [Chen et al., NeurIPS 2017]
**[Direct PDF (ArXiv)](https://www.google.com/search?q=https://arxiv.org/pdf/1711.07752.pdf)**
* **[20] Structured KD for Semantic Segmentation** [Liu et al., CVPR 2019]
**[Direct PDF (ArXiv)](https://www.google.com/search?q=https://arxiv.org/pdf/1903.04197.pdf)**

### **TinyML Insight:**

If you're wondering why **Mixup [23]** and **Cutout [22]** are in a TinyML syllabus: smaller models (student models) often suffer from high bias. These regularization techniques allow you to train them for much longer on the same dataset without the model simply "memorizing" the noise, which is critical for getting high accuracy out of a very small parameter count.

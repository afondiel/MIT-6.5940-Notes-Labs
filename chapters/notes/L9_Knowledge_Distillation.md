# Lecture 9: Knowledge Distillation

- **Lecturers:** Professor Song Han
- **Date:** Fall 2023
- **Corresponding Course Website Section:** efficientml.ai

## 1. 🎯 Why It Matters for Edge AI

* **The Core Problem:** Small, efficient "student" models (often the output of Pruning or NAS) lack the capacity to be trained from scratch to match the performance of large, complex "teacher" models. They struggle to capture the full nuances of the data.
* **Edge AI Benefits:** **Knowledge Distillation (KD)** transfers the rich, learned information (the "dark knowledge") from a high-performance, large **Teacher model** to the efficient **Student model**. This allows the small model to achieve an accuracy that is often much higher than it would reach if trained traditionally, maximizing the performance of the tiny network deployed on the edge.

---

## 2. 📝 Key Concepts and Theory

* **Definition & Overview:** KD is a training paradigm where a compact Student model is trained to mimic the output of a larger, more powerful Teacher model.
* **The KD Loss Function:** The training objective for the Student model $S$ consists of two main components:
    $$\mathcal{L}_{\text{KD}} = (1 - \alpha) \mathcal{L}_{\text{hard}} + \alpha \mathcal{L}_{\text{soft}}$$
    * **Hard Loss ($\mathcal{L}_{\text{hard}}$):** The traditional cross-entropy loss between the Student's prediction and the **true (hard) label**.
    * **Soft Loss ($\mathcal{L}_{\text{soft}}$):** The distillation loss (e.g., KL divergence) between the **soft targets** (Teacher's output logits) and the Student's soft predictions.
* **Soft Targets and Temperature ($T$):** Soft targets are the normalized probability distributions (logits) of the Teacher model, softened by a **Temperature hyperparameter ($T$)**:
    $$P_i^{\text{soft}} = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}$$
    * **High $T$:** Produces a smoother probability distribution, revealing the **"dark knowledge"**—the relative probabilities the Teacher assigns to incorrect classes. This is the valuable information transferred to the Student.
    * **Low $T$ (or $T=1$):** Approximates the standard softmax function.
* **Types of Knowledge Transfer:**
    1.  **Response-based KD (Output Distillation):** Using only the final logits (soft targets).
    2.  **Feature-based KD (Hint Learning):** Using intermediate layer outputs (feature maps) from the Teacher to guide the Student's hidden layers.
    3.  **Relation-based KD:** Transferring relationships between data points or network layers.

---

## 3. ⚙️ Practical Implementation & Tools

* **Implementation Steps:**
    1.  **Train the Teacher:** Ensure the Teacher model is fully converged to a high accuracy.
    2.  **Initialize the Student:** Use the target efficient model (pruned, NAS-found, or hand-designed MobileNet).
    3.  **Distillation Training:** Use the combined loss function $\mathcal{L}_{\text{KD}}$ for training. Key hyperparameter tuning includes the **Temperature $T$** (often set between 2 and 4) and the **weight $\alpha$** (balancing hard vs. soft loss, often $0.5 < \alpha < 0.9$).
* **Tools:**
    * **Hugging Face Accelerate/Trainer:** Includes distillation recipes and utilities.
    * **PyTorch/TensorFlow:** Standard tensor operations are used to implement the loss functions, with custom training loops often required to handle the Teacher's forward pass.

---

## 4. ⚖️ Trade-offs and Real-World Impact

* **Trade-off:** KD requires performing two forward passes (Teacher and Student) during the Student's training, slightly increasing training time. However, this cost is often negligible compared to the massive gain in final deployment accuracy.
* **Synergy with Other Techniques:** KD is highly synergistic with **NAS** (training the found optimal architecture) and **Pruning/Quantization** (recovering accuracy loss).
* **The Bottleneck:** The primary difficulty lies in finding the optimal hyperparameters ($T, \alpha$) and, for feature-based KD, deciding which intermediate layers of the Teacher and Student should be matched.

---

## 5. 🧪 Hands-on Lab Preview

* **What you will do:** Implement a basic **Response-based Knowledge Distillation** setup. You will train a small CNN (Student) using a large ResNet (Teacher) and demonstrate that the distilled Student achieves significantly higher accuracy than a Student trained only on the hard labels.
* **Key Skill Acquired:** Defining and optimizing the $\mathcal{L}_{\text{KD}}$ loss function and effectively using a high temperature $T$ to transfer the Teacher's valuable "dark knowledge."

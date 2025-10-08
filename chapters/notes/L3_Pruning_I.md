# Lecture 3: Pruning and Sparsity (Part I)

- **Lecturers:** Professor Song Han
- **Date:** Fall 2023
- **Corresponding Course Website Section:** efficientml.ai

## 1. 🎯 Why It Matters for Edge AI

* **The Core Problem:** Large, over-parameterized models (often necessary for achieving high accuracy) have millions or billions of weights. Many of these weights are redundant or contribute very little to the final prediction. This bloats model size and memory requirements.
* **Edge AI Benefits:** **Pruning** creates a **sparse model**, drastically reducing the memory footprint (crucial for microcontrollers) and potentially the computation (by skipping zero-valued operations). This enables fitting large-capacity models onto tiny devices and reducing off-chip memory access time.

---

## 2. 📝 Key Concepts and Theory

* **Definition & Overview:** Pruning is the technique of removing redundant weights (or neurons/filters) from a trained neural network, resulting in a model with fewer parameters and FLOPs.
* **The Pruning Pipeline (The "3-Step" Process):**
    1.  **Train:** Train the dense (full-size) model to convergence.
    2.  **Prune:** Determine which weights/structures to remove based on a **pruning criterion** (e.g., magnitude).
    3.  **Fine-tune (Retrain):** Retrain the resulting sparse network on the dataset to recover lost accuracy.
* **Unstructured Pruning (The simplest form):**
    * **Mechanism:** Individual, insignificant weights are set to zero, creating a sparse weight matrix.
    * **Criterion (Magnitude Pruning):** Remove weights $w_i$ with the smallest absolute magnitude.
        $$\text{Prune if } |w_i| < \text{Threshold}$$
    * **Pros:** Achieves the highest sparsity and parameter reduction with minimal accuracy loss.
    * **Cons (The Hardware Problem):** The sparsity pattern is *random*. Standard dense hardware (GPUs, CPUs) often cannot efficiently skip these scattered zeros, making the theoretical FLOPs reduction negligible in practice. This leads to poor real-world speedup without specialized **sparse hardware** or a highly efficient sparse runtime.

---

## 3. ⚙️ Practical Implementation & Tools

* **Implementation Steps:**
    1.  **Select Criterion:** Use weight magnitude as the simplest and most effective starting point.
    2.  **Determine Sparsity:** Choose a target percentage of weights to remove (e.g., 50%, 90%).
    3.  **Apply Mask:** Create a binary mask (0 for pruned, 1 for kept) and multiply it with the weight tensor.
    4.  **Fine-tune:** Use a low learning rate to retrain the unpruned weights while keeping the pruned weights at zero.
* **Tools:**
    * **PyTorch's `torch.nn.utils.prune`:** Provides a modular API for implementing various pruning methods.
    * **TensorFlow Model Optimization Toolkit:** Includes modules for sparsity and pruning techniques.

---

## 4. ⚖️ Trade-offs and Real-World Impact

* **Trade-off:** High sparsity (e.g., 90%) leads to a minimal memory footprint but usually requires longer fine-tuning to recover accuracy, and the speedup is **not guaranteed** on standard edge processors due to **irregular memory access** patterns.
* **The "Lottery Ticket Hypothesis" (LTH):** The observation that within a large, randomly initialized network, a small **"winning ticket"** (a subnetwork) exists that, when trained in isolation, can achieve comparable accuracy to the full network. This justifies the pruning process.
* **Hardware Bottleneck:** The primary reason Unstructured Pruning is primarily a **memory reduction** technique on generic hardware; achieving **latency reduction** requires moving to **Structured Pruning** (Lecture 4).

---

## 5. 🧪 Hands-on Lab Preview

* **What you will do:** Implement **Iterative Magnitude Pruning** on a small CNN (like LeNet or MobileNet block). You will apply the pruning mask and observe the recovery of accuracy during fine-tuning.
* **Key Skill Acquired:** Implementing the core **pruning-fine-tune loop** and calculating the effective **model compression ratio** (sparse size / dense size).

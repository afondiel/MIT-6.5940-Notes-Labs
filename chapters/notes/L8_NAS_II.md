# Lecture 8: Neural Architecture Search (Part II)

- **Lecturers:** Professor Song Han
- **Date:** Fall 2023
- **Corresponding Course Website Section:** efficientml.ai

## 1. 🎯 Why It Matters for Edge AI

* **The Core Problem:** The prohibitive search cost of traditional NAS (training thousands of models) is the single biggest blocker to widespread adoption. A faster, more efficient search method is needed.
* **Edge AI Benefits:** **One-Shot NAS** (especially the **Once-for-All** approach from the HAN Lab) drastically reduces search time by eliminating the need to train individual models. This makes finding the optimal, hardware-specialized architecture feasible within minutes or hours, rather than weeks.

---

## 2. 📝 Key Concepts and Theory

* **One-Shot NAS:** The concept of training a single, massive **"Supernet"** that contains all possible candidate architectures as subnetworks.
    * **Search:** The search is simplified to finding the best *path* or *subnetwork* within the already trained Supernet.
    * **No Retraining:** Candidate subnetworks can inherit weights directly from the Supernet, avoiding the need for full training.
* **Once-for-All (OFA) NAS:** A specific, highly effective One-Shot NAS approach:
    1.  **Train the OFA Supernet:** Train the large Supernet with a specialized training method (e.g., sandwich rule) so that all its subnetworks (with varying depth, width, and kernel size) perform well.
    2.  **Decouple Search and Training:** The search phase now only requires evaluating the *accuracy* and *latency* of subnetworks without any further training.
    3.  **Hardware-Aware Search:** Use an evolutionary search algorithm to explore the subnetworks, prioritizing those that satisfy a specific latency constraint on a given target device.
* **Gradient-Based NAS (DARTS):** A method that makes the architecture search space *continuous* by defining architecture weights and optimizing them simultaneously with the model weights using gradient descent. This is another way to speed up the search dramatically.

---

## 3. ⚙️ Practical Implementation & Tools

* **Implementation Steps (OFA):**
    1.  **Define Architecture Space:** Specify the ranges for depth, width, and kernel sizes.
    2.  **Supernet Training:** Train the largest architecture (Supernet) with a specialized sampler to ensure weight sharing works effectively for all subnetworks.
    3.  **Latency Prediction:** Build a **latency lookup table** or a **latency predictor model** for the target edge device (e.g., Raspberry Pi, Jetson Nano).
    4.  **Evolutionary Search:** Use the predictor and the Supernet's inherited accuracy to quickly find the Pareto-optimal subnetworks.
* **Tools:**
    * **Once-for-All (OFA) Framework:** The specific open-source code/tool from the MIT HAN Lab that implements this framework.
    * **Neural Network Latency Predictors:** Tools that estimate the runtime of a model on specific hardware, which are essential for hardware-aware NAS.

---

## 4. ⚖️ Trade-offs and Real-World Impact

* **Search Cost Reduction:** OFA reduces the total NAS cost from **thousands of GPU days** (traditional) to **single-digit days** for the Supernet training plus **minutes/hours** for the hardware-aware search.
* **Deployment:** OFA enables deploying different, highly efficient subnetworks from the same trained Supernet onto a *variety* of edge devices, a capability known as **Specialized Deployment**. For example, one subnetwork can go to an MCU, and a slightly larger one to a mobile phone.
* **Challenge:** The Supernet training phase is complex and memory-intensive, requiring advanced techniques to manage the training of all possible configurations simultaneously.

---

## 5. 🧪 Hands-on Lab Preview

* **What you will do:** Utilize the pre-trained **OFA Supernet**. You will input latency constraints for different hypothetical edge devices and use the evolutionary search algorithm to quickly generate and test several highly optimized subnetworks for those constraints.
* **Key Skill Acquired:** Performing **hardware-aware architecture search** and selecting the optimal architecture on the **Pareto curve** (the line connecting the most accurate models for a given latency/FLOPs).

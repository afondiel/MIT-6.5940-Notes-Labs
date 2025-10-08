# Lecture 7: Neural Architecture Search (Part I)

- **Lecturers:** Professor Song Han
- **Date:** Fall 2023
- **Corresponding Course Website Section:** efficientml.ai

## 1. 🎯 Why It Matters for Edge AI

* **The Core Problem:** Human-designed networks (like ResNet or VGG) are often sub-optimal for specific tasks or target hardware. Finding a specialized, high-performance architecture is a time-consuming, trial-and-error process for human engineers.
* **Edge AI Benefits:** **NAS** automates the design of specialized networks, allowing us to discover models that are not only highly accurate but also highly efficient (low FLOPs, high latency) for a specific **target device** (e.g., a phone's NPU or a specific MCU). This leads to the best **hardware-aware** performance.

---

## 2. 📝 Key Concepts and Theory

* **Definition & Overview:** Neural Architecture Search (NAS) is the process of using automated search algorithms (e.g., Reinforcement Learning, Evolutionary Algorithms, Gradient Descent) to find the optimal neural network architecture for a given task and efficiency constraint.
* **The Three Components of NAS:** A NAS system is defined by three main parts:
    1.  **Search Space ($\mathcal{A}$):** Defines the set of all possible architectures that can be generated. This could range from simple block-level operations (e.g., which convolution kernel size to use) to entire macro-structures.
    2.  **Search Strategy ($\mathcal{S}$):** The algorithm used to explore the search space (e.g., Random Search, RL, Evolution, Gradient-based methods).
    3.  **Performance Estimation Strategy ($\mathcal{E}$):** The method for evaluating a candidate architecture's quality, typically measured by **Accuracy** and **Latency/FLOPs**.
* **Challenges of Traditional NAS:** The search space is often prohibitively large, and evaluating each candidate architecture by fully training it is extremely computationally expensive (e.g., thousands of GPU hours).

---

## 3. ⚙️ Practical Implementation & Tools

* **Search Space Design (Common):**
    * **Cell-based Search:** Designing a small "cell" (e.g., a convolutional block) and stacking it repeatedly. This greatly restricts the search space while maintaining good performance.
    * **Macro Search:** Searching for the overall layer sequence and connectivity.
* **Search Strategies:**
    * **Evolutionary Algorithms (EA):** Treating architectures as "individuals," subjecting them to mutation and crossover, and keeping the "fittest" (most accurate/efficient) for the next generation.
    * **Reinforcement Learning (RL-based NAS):** Using an RNN "controller" to generate the architectural choices (actions) and rewarding the controller based on the child network's performance.

---

## 4. ⚖️ Trade-offs and Real-World Impact

* **Trade-off:** NAS requires a massive initial computational investment (**search cost**) to find the optimal architecture, but the resulting model (**target cost**) is often much smaller and faster than a hand-designed one. This initial cost is often worth it for widely deployed models (e.g., in mobile phones).
* **Hardware Awareness:** The **Performance Estimator** in modern NAS should explicitly measure or predict the latency of the candidate model on the **actual target hardware** (e.g., a specific mobile phone chip) rather than just relying on proxy metrics like FLOPs. This is **Hardware-Aware NAS**.
* **Transferability:** Architectures found via NAS are highly specialized. They may not perform optimally if transferred to a different task or hardware platform.

---

## 5. 🧪 Hands-on Lab Preview

* **What you will do:** Explore a simplified, toy NAS search space. You will define a simple search space and use a Random Search strategy to find an architecture that minimizes FLOPs while achieving a minimum required accuracy on a small dataset.
* **Key Skill Acquired:** Defining a **search space** and understanding how to formulate the **objective function** to balance both accuracy and efficiency constraints ($\text{Objective} = \text{Accuracy} + \lambda \cdot \log(\text{Latency})$, where $\lambda$ is a weight).

# **Lesson 23: Quantum Machine Learning (Part II)** 

provides an in-depth look at Parameterized Quantum Circuits (PQCs), their training methodologies, and frameworks for optimizing quantum architectures in the presence of noise.

### **I. Parameterized Quantum Circuits (PQC)**
PQCs are quantum circuits that combine **fixed gates** with **parameterized gates**, and they are fundamental to hybrid classical-quantum models like Quantum Neural Networks (QNNs).
*   **Expressivity:** This metric measures how well a quantum circuit covers the **Hilbert space** by observing how states deviate from a uniform distribution.
*   **Entanglement Capability:** Measured using the **Meyer-Wallach measure** (ranging from 0 to 1), this indicates a circuit's ability to generate entangled states.
*   **Hardware Efficiency:** A critical design consideration that evaluates whether the PQC respects **qubit connectivity** and utilizes the hardware’s **native gates**.
*   **Data Encoding:** Classical data is mapped into the quantum domain using various techniques:
    *   **Basis Encoding:** Represents data in binary form, similar to classical machines.
    *   **Amplitude Encoding:** Encodes data into the **statevector**; it is efficient as $N$ features require only $\log N$ qubits.
    *   **Angle Encoding:** Uses data values as the **rotation angles** for qubit gates.
    *   **Arbitrary Encoding:** Employs an arbitrary PQC where input data serves as rotation angles.

### **II. PQC Training and Challenges**
Training involves optimizing parameters within the PQC to perform data-driven tasks.
*   **Gradient Computation:** 
    *   **Finite Difference:** A general method that approximates gradients by perturbing parameters by a small $\epsilon$.
    *   **Parameter Shift Rule:** Calculates gradients by running the circuit with positive and negative shifts on the quantum device.
    *   **SPSA (Simultaneous Perturbation Stochastic Approximation):** Simultaneously perturbs all dimensions, achieving similar convergence to standard gradient descent with significantly fewer circuit runs.
*   **Barren Plateaus:** A major obstacle where gradients "vanish" (variance reduces exponentially) as the circuit scale increases, making training difficult.

### **III. Quantum Classifiers and Tasks**
*   **Quantum Neural Networks (QNN):** These classifiers encode pixels as angles, process them through trainable quantum layers, and perform measurements followed by a **Softmax** function for classification.
*   **Other Tasks:** PQCs are also applied to the **Variational Quantum Eigensolver (VQE)** and the **Quantum Approximate Optimization Algorithm (QAOA)**.

### **IV. Noise-Aware On-Chip Training (QOC)**
Quantum noise significantly degrades the reliability of gradients computed on real hardware. 
*   **Noise Impact:** Small magnitude gradients often suffer from large relative errors.
*   **Gradient Pruning:** Techniques like **Probabilistic Pruning** can remove these unreliable gradients, reducing the performance gap between classical simulations and real quantum devices.
*   **Scalability:** On-chip training scales better than classical simulations, which face exponential memory and runtime costs as the number of qubits increases.

### **V. TorchQuantum Library**
**TorchQuantum** is a PyTorch-native library designed to enable machine learning-assisted, hardware-aware quantum algorithm design.
*   **Core Features:** It supports **GPU-accelerated** statevector simulation, automatic gradient computation, and easy construction of hybrid classical-quantum networks.
*   **Integration:** It provides tools to convert models to other frameworks, such as **IBM Qiskit**.

### **VI. Quantum Architecture Search (QuantumNAS)**
Given the large design space and the impact of noise, automated search for robust architectures is essential.
*   **Framework:** **QuantumNAS** constructs a **SuperCircuit** where candidate **SubCircuits** share and inherit parameters, allowing for efficient search without training every candidate from scratch.
*   **Noise-Adaptive Search:** An evolutionary strategy searches for the best SubCircuit and its **qubit mapping** on the target hardware.
*   **Iterative Pruning:** Small-magnitude quantum gates (angles close to 0) are iteratively pruned and the remaining parameters are fine-tuned to maintain reliability.
*   **Alternative Frameworks:** Other methods include using **Reinforcement Learning**, **Differentiable Search**, or **Graph Transformers** for fidelity estimation.

## References
- Source: https://notebooklm.google.com/notebook/39b7bd5c-37ef-451c-83dc-98b62ec8dc04
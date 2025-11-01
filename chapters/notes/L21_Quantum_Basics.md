# Lecture 21: Basics of Quantum Computing ⚛️

## Quick Reference

|Item|Reference|
|---|---|
| Slides | [View Slides](https://drive.google.com/drive/folders/1A3P6IBuS8wCzLlpdRiQBO9b1uoK3pnPf?usp=sharing)|
| Video | [EfficientML.ai Lecture 21](http://www.youtube.com/watch?v=6cAmS-_vEh8)  |
|Lab| -- |
|Professor|Hanrui Wang|


## 1. 🎯 Why It Matters for Efficient ML

* **The Core Problem:** Classical machine learning is reaching theoretical limits on resource-intensive tasks like molecular simulation, materials science, and large-scale optimization. Certain problems scale exponentially with classical resources.
* **Efficient ML Benefits:** Quantum computing offers a fundamentally different computational model. It has the **potential for exponential speedups** on specific, difficult problems (e.g., Shor's algorithm for factoring, Grover's algorithm for search) that may be leveraged to accelerate or enhance future ML tasks. This new architecture represents a significant step in the pursuit of **ultimate computational efficiency**.

---

## 2. 📝 Key Concepts and Theory

* **The Qubit (Quantum Bit):**
    * **Classical Bit:** Stores information as 0 or 1.
    * **Qubit ($|\psi\rangle$):** Stores information as a **superposition** of $|0\rangle$ and $|1\rangle$. The state is a vector in a 2D complex space, represented as a linear combination: $|\psi\rangle = \alpha|0\rangle + \beta|1\rangle$, where $\alpha$ and $\beta$ are complex amplitudes and $|\alpha|^2 + |\beta|^2 = 1$. The state can be visualized on the **Bloch Sphere**. 
* **Superposition:** The ability of an $n$-qubit system to exist in up to $2^n$ different states simultaneously. This grants quantum computers an **exponential capacity** to store and process information compared to $n$ classical bits.
* **Entanglement:** A unique quantum correlation where the state of two or more qubits is inextricably linked. They share a single, combined quantum state. Entanglement is a key **computational resource** that allows quantum algorithms to explore solutions far beyond the capabilities of classical systems.
* **Quantum Gates and Circuits:**
    * **Quantum Gates:** Unitary (reversible) transformations applied to qubit states, represented by unitary matrices. They are the building blocks of quantum algorithms.
    * **Key Gates:** $\mathbf{Hadamard (H)}$ (creates superposition), $\mathbf{Pauli-X}$ (NOT gate), $\mathbf{CNOT}$ (Controlled-NOT, essential for creating entanglement).
    * **Quantum Circuit:** A sequence of quantum gates applied to a register of qubits over time.
* **Measurement:** The process that forces a qubit out of superposition and collapses it into a single classical state (0 or 1) with probability $|\alpha|^2$ or $|\beta|^2$. This is the moment the quantum state interacts with the classical world.

---

## 3. ⚙️ Practical Implementation & Architectures

* **Gate-Based Quantum Computing:** The dominant model (used by IBM, Google, Rigetti). Algorithms are compiled into a sequence of single- and two-qubit gates.
* **Quantum Hardware Implementations:**
    * **Superconducting Circuits:** Uses Josephson junctions (IBM, Google). Requires cryogenic temperatures.
    * **Trapped Ions:** Uses lasers to suspend and control individual ions (IonQ). Known for high-fidelity gates.
    * **Photonic Systems:** Uses photons as qubits (PsiQuantum, Xanadu).
* **Programming Tools:**
    * **Qiskit (IBM):** An open-source framework for working with quantum computers at the level of circuits, pulses, and algorithms.
    * **Cirq (Google):** A framework for creating, editing, and invoking NISQ circuits.

---

## 4. ⚖️ Trade-offs and Real-World Impact

* **Trade-off (Power vs. Noise):** Quantum computation offers potential **exponential power**, but current devices (NISQ) are extremely **noisy** and **error-prone**. This noise limits the complexity and depth of circuits that can be executed reliably.
* **Impact:** While commercial quantum computers are not yet faster than classical ones for general ML, the field is rapidly advancing. Quantum computing has already driven breakthroughs in theoretical computer science and could eventually transform fields like drug discovery (simulation) and financial modeling (optimization).
* **The Critical Bottleneck:** **Coherence Time** (how long a qubit can maintain its quantum properties) and **Gate Fidelity** (the accuracy of quantum operations) are the biggest bottlenecks today. These issues are directly addressed in later lectures on noise robustness.

---

## 5. 🧪 Hands-on Lab Preview

* **What you will do:** Use an open-source quantum simulation library (like Qiskit or Pennylane) to **build a simple quantum circuit** with a Hadamard and CNOT gate. You will run the circuit and analyze the measurement probabilities to demonstrate the creation of **superposition and entanglement**.
* **Key Skill Acquired:** Understanding the fundamental programming model of quantum circuits and physically interpreting the results of a quantum measurement.

***


## References

- EfficientML.ai Course | 2023 Fall | MIT 6.5940: [ Complete course video series ](https://youtube.com/playlist?list=PL80kAHvQbh-pT4lCkDT53zT8DKmhE0idB&si=Uu00N0zKopEixhw3).
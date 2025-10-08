# Lecture 10: MCUNet: TinyML on Microcontrollers

- **Lecturers:** Professor Song Han
- **Date:** Fall 2023
- **Corresponding Course Website Section:** efficientml.ai

## 1. 🎯 Why It Matters for Edge AI

* **The Core Problem:** Microcontrollers (MCUs) have extremely limited resources, often featuring **KB-level SRAM** (e.g., 20KB) and minimal flash memory. This is fundamentally incompatible with standard deep learning, where even a small MobileNet can require MBs of memory. Existing TinyML solutions often require *hand-tuning* the model for each specific MCU.
* **Edge AI Benefits:** **MCUNet** is a system-algorithm co-design approach that enables running deep learning on off-the-shelf MCUs. It automatically co-optimizes the neural architecture and the inference library, pushing the boundary of what's possible in **TinyML**. It enables deploying ImageNet-level models on tiny devices.

---

## 2. 📝 Key Concepts and Theory

* **Definition & Overview (MCUNet):** An end-to-end framework combining an efficient neural network architecture with a lightweight inference engine specifically designed for ultra-low-power microcontrollers.
* **Neural Architecture Search for MCUs (TinyNAS):**
    * **Constraint:** The search must explicitly prioritize extremely low memory (SRAM) usage, not just FLOPs.
    * **Output:** A specialized family of neural networks optimized for KB-level SRAM constraints.
* **Memory-Efficient Inference Engine (TinyEngine):**
    * **The Challenge:** The biggest bottleneck is often not the computation, but the frequent movement of large activation tensors between the tiny on-chip SRAM and the much larger, but slower, flash memory.
    * **Key Innovation (SRAM Allocation):** TinyEngine uses a **layer-by-layer memory planning** strategy, breaking down large layers into smaller, sequential tasks that reuse the small SRAM area as much as possible, minimizing costly off-chip flash access. This includes sophisticated **tensor tiling** strategies.

---

## 3. ⚙️ Practical Implementation & Tools

* **Implementation Steps (MCUNet Flow):**
    1.  **Model Selection/Search (TinyNAS):** Find an architecture that meets the SRAM constraint for activations and model size.
    2.  **Quantization:** Apply aggressive quantization (often INT8 or even lower) to the model weights.
    3.  **TinyEngine Compilation:** The model and the memory planning are fed into the TinyEngine compiler. The compiler generates **C code** with specific memory allocation instructions for the target MCU.
    4.  **Deployment:** The C code is flashed onto the MCU device.
* **Tools:**
    * **TinyML Runtimes (e.g., TFLite Micro):** While TFLite Micro provides a runtime, specialized engines like **TinyEngine** go deeper into co-design for memory efficiency.
    * **Vendor Specific Compilers:** Tools like the ones from Arm or other MCU vendors for bare-metal C/C++ compilation.

---

## 4. ⚖️ Trade-offs and Real-World Impact

* **Trade-off:** The high level of co-optimization means the resulting solution is **highly hardware-specific**. The TinyEngine generated for one MCU might not be optimal for another due to differences in SRAM size, cache, and flash layout.
* **Impact:** MCUNet demonstrated a significant breakthrough, enabling complex tasks like ImageNet classification on commercial off-the-shelf microcontrollers (e.g., an STM32F746) with minimal memory, proving that AI is feasible on devices with only a few hundred KB of memory.
* **The Critical Bottleneck:** In TinyML, **SRAM/Activation Memory** is the absolute most constrained resource. Solutions must prioritize minimizing the maximum size of *any* activation tensor during inference.

---

## 5. 🧪 Hands-on Lab Preview

* **What you will do:** Work with a **TinyEngine-like simulator** or framework. You will take a tiny, pre-quantized model and analyze its **memory access profile**, demonstrating how a good memory scheduling strategy can reduce the peak SRAM usage to fit a target MCU's constraints.
* **Key Skill Acquired:** Understanding memory access patterns in embedded devices and appreciating the importance of **system-algorithm co-design** for ultra-constrained environments.
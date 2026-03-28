# **Lesson 10: MCUNet and TinyML** 

focuses on the challenges of bringing deep learning to resource-constrained Internet of Things (IoT) devices and introduces the **MCUNet** framework as a solution through system-algorithm co-design.

### **I. The Challenges of TinyML**
The primary obstacle for TinyML is the extreme memory constraint of microcontrollers compared to cloud or mobile platforms.
*   **Memory Bottlenecks:** While Cloud AI has tens of gigabytes of memory, microcontrollers often have less than **1MB of Flash** (for weight storage) and **320KB of SRAM** (for activations). 
*   **Activation Constraints:** In CNN inference, peak SRAM usage (the sum of input and output activations for a layer) is typically the bottleneck. Existing mobile-optimized models like MobileNetV2 reduce parameter size but fail to significantly reduce peak activation memory, often exceeding microcontroller limits by 5x or more.

### **II. MCUNet: System-Algorithm Co-design**
MCUNet addresses these limits by jointly optimizing the neural architecture and the execution engine.
*   **TinyNAS (Neural Architecture Search):** A two-stage automated design process. 
    *   **Search Space Optimization:** Instead of using a fixed mobile search space, TinyNAS first identifies an optimized design space (resolution and width multiplier) that fits the specific device's memory/storage constraints while maximizing model capacity (FLOPs).
    *   **Model Specialization:** It then searches for specific sub-networks within a "super-network" using weight sharing, allowing one-shot design for diverse hardware platforms.
*   **TinyEngine:** A specialized compiler and runtime that generates efficient code to execute the models found by TinyNAS.

### **III. MCUNetV2: Breaking the Memory Bottleneck**
MCUNetV2 introduces **Patch-based Inference** to further reduce SRAM usage by up to 8x.
*   **Concept:** Standard inference processes models layer-by-layer, which requires storing entire feature maps in SRAM. Patch-based inference processes only a small spatial portion (patch) of the initial high-resolution layers at a time, drastically lowering peak memory.
*   **Network Redistribution:** To avoid the computational overhead caused by overlapping receptive fields in patches, MCUNetV2 redistributes the network architecture to use smaller receptive fields in early layers where patch-based inference is applied.

### **IV. Key Applications and Results**
*   **Tiny Vision:** MCUNet enabled the first ImageNet-level classification on commercial microcontrollers, achieving **>70% accuracy**. It also enables **Visual Wake Words** (e.g., detecting a person to wake up a larger system) on devices with as little as 32KB of SRAM.
*   **Object Detection:** By fitting larger input resolutions through patch-based inference, MCUNetV2 enabled tasks like face/mask detection and person detection on microcontrollers.
*   **Tiny Audio and Time-Series:** The framework also provides significant speedups and memory savings for speech command recognition and anomaly detection in manufacturing or healthcare.

## References
- Source: https://notebooklm.google.com/notebook/39b7bd5c-37ef-451c-83dc-98b62ec8dc04
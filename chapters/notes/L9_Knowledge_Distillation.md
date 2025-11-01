# Lecture 9: Knowledge Distillation

## Quick Reference

|Item|Reference|
|---|---|
| Slides | [View Slides](https://drive.google.com/drive/folders/1A3P6IBuS8wCzLlpdRiQBO9b1uoK3pnPf?usp=sharing)|
| Video | [EfficientML.ai Lecture 9 - Knowledge Distillation](https://www.youtube.com/watch?v=dSDW_c789zI)  |
|Lab| -- |
|Professor|[Song Han](https://github.com/songhan)|

## **1. Introduction to Knowledge Distillation (KD)**

Knowledge Distillation (KD) is a model compression technique where a small, efficient network, known as the **Student**, is trained to mimic the behavior and knowledge of a larger, high-performing network, the **Teacher**.

### **A. Core Motivation**

* **Small Models are Hard to Train:** Shrinking a model via pruning or NAS often results in a "student" model that is difficult to train from scratch to the same level of accuracy as the large "teacher" model.  
* **The Teacher's "Knowledge":** The teacher model contains valuable knowledge beyond just its final hard predictions (the one-hot labels). The distribution of class probabilities (the "soft targets") provides richer information about the data relationships.

### **B. Standard KD Framework**

The Student model is trained with a modified loss function that combines two components:

1. **Soft Target Loss (Distillation Loss):** Measures the divergence between the Student's soft predictions and the Teacher's soft predictions, using the **Kullback-Leibler (KL) Divergence** loss.  
   * This loss encourages the Student to learn the relationships and ambiguities the Teacher learned (e.g., if the Teacher thinks "cat" has a 90% probability and "dog" has a 10% probability, the Student learns that relationship, even if the true label is "cat").  
   * **Temperature Parameter (**$\\tau$**):** A hyperparameter applied to the softmax function for both Teacher and Student. A higher $\\tau$ creates softer (more uniform) probability distributions, revealing more nuance in the Teacher's knowledge.  
2. **Hard Target Loss (Standard Cross-Entropy Loss):** Measures the divergence between the Student's hard predictions and the true one-hot ground-truth labels.

$$
\text{Total Loss} = \alpha \cdot \text{KL}(\text{Student}\_{\tau}, \text{Teacher}_{\tau}) + (1-alpha) \cdot \text{CE}(\text{Student}, \text{Ground Truth})
$$

Where $\\alpha$ is a blending factor.

## **2\. Advanced Distillation Techniques**

Beyond simply matching the output probabilities, advanced KD methods seek to transfer knowledge from intermediate layers or the training process itself.

### **A. Feature Distillation (Hinton, FSP)**

Instead of matching the output logits, the Student is guided to match the feature maps or hidden layer activations of the Teacher.

* **Process:** A loss term (e.g., L2 distance) is added to minimize the difference between the intermediate feature map of the Teacher and a corresponding, *re-scaled* feature map of the Student.  
* **Benefit:** This forces the Student to learn the internal representations and feature extraction capabilities of the Teacher, rather than just the final output mapping.

### **B. Transferring Knowledge via Data (Data-Free Distillation)**

In many real-world scenarios, the original training data used for the Teacher is not available (due to privacy or logistical issues).

* **Data-Free Knowledge Distillation (DFKD):** A generator network is trained to synthesize "hard-to-learn" or "challenging" images/data points that maximize the difference between the Student's output and the Teacher's output.  
* **Process:** The Student and Teacher are then distilled using this synthesized data. The generator effectively creates a small, highly informative dataset for distillation.  
* **Benefit:** Enables model compression and specialization without requiring access to sensitive or large proprietary datasets.

### **C. Self-Distillation**

This technique uses a single model architecture and trains it to distill knowledge into itself.

* **Concept:** A single network can be viewed as its own Teacher and Student. The "Teacher" might be an exponential moving average (EMA) of the Student's weights during training, or different parts of the network can distill into other parts.  
* **Benefit:** Provides a strong regularization effect and often improves the final accuracy of the single target model without requiring an external large model.

## **3\. Summary of Model Compression Techniques (Part I)**

| Technique | Goal | Primary Benefit | Implementation Cost | Hardware Acceleration |
| :---- | :---- | :---- | :---- | :---- |
| **Pruning** (L3, L4) | Reducing redundant parameters/FLOPs | Shrinks model size, potentially speedup | Medium (requires iterative training/fine-tuning) | Structured: High. Unstructured: Low (needs special hardware). |
| **Quantization** (L5, L6) | Reducing numerical precision | Reduced memory access cost, faster integer computation | Medium (requires calibration or QAT) | High (INT8/INT4 optimized hardware) |
| **NAS** (L7) | Designing efficient architectures | Finds optimal design point (Accuracy vs. Latency) | High (search is expensive) | High (NAS designs hardware-aware models) |
| **Distillation** (L8) | Transferring knowledge to small model | Recovers accuracy lost during compression/shrinking | Low-to-Medium (simple loss function change) | Indirectly: Enables the use of smaller, faster models. |

These notes complete the first major section, **Efficient Inference: Model Compression**. The next part of the course typically moves into domain-specific optimizations, such as for the Transformer architecture.
## References

- EfficientML.ai Course | 2023 Fall | MIT 6.5940: [ Complete course video series ](https://youtube.com/playlist?list=PL80kAHvQbh-pT4lCkDT53zT8DKmhE0idB&si=Uu00N0zKopEixhw3).
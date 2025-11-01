# Lecture 20: Efficient Fine-tuning and Prompt Engineering

## Quick Reference

|Item|Reference|
|---|---|
| Slides | [View Slides](https://drive.google.com/drive/folders/1A3P6IBuS8wCzLlpdRiQBO9b1uoK3pnPf?usp=sharing)|
| Video | [EfficientML.ai Lecture 20](http://www.youtube.com/watch?v=6cAmS-_vEh8)  |
|Lab| -- |
|Professor|[Song Han](https://github.com/songhan)|


## 1. 🎯 Why It Matters for Edge AI

* **The Core Problem:** Large Language Models (LLMs) and other foundation models are immense (Billions to Trillions of parameters). While powerful, **full fine-tuning** is **prohibitively expensive** (costly, slow, requires massive storage) and often unnecessary since most new tasks only require adapting a small portion of the model's knowledge.
* **Edge AI Benefits:** Efficient tuning methods allow practitioners to **specialize models** for proprietary datasets or specific user tasks with minimal resources. **Parameter-Efficient Fine-Tuning (PEFT)** and **Prompt Engineering** enable rapid experimentation, personalization, and deployment in resource-constrained or distributed environments.

---

## 2. 📝 Key Concepts and Theory

* **Definition & Overview (Customization Methods):**
    * **Full Fine-tuning (FFT):** Trains all parameters. High performance, high cost.
    * **Prompt Engineering:** Trains 0 parameters. Low cost, limited specialization (relies on in-context learning).
    * **PEFT:** Trains a very small number of parameters. Great cost-performance trade-off.
* **Prompt Engineering Techniques:**
    * **In-Context Learning (ICL):** Providing examples directly in the input prompt (Few-Shot).
    * **Chain-of-Thought (CoT) Prompting:** Guiding the model to output **intermediate reasoning steps** before the final answer, dramatically boosting performance on complex reasoning and arithmetic.
* **Parameter-Efficient Fine-Tuning (PEFT):**
    * **LoRA (Low-Rank Adaptation):** Decomposes the update matrix $\Delta W$ for pre-trained weights $W$ into two small, low-rank matrices ($B \times A$). Only the parameters in $A$ and $B$ are trained. $\Delta W = B A$. This typically reduces trainable parameters by $\mathbf{1000 \times}$ while maintaining high performance.
    * **Prompt Tuning:** Freezes the entire LLM and only optimizes a small set of **soft, continuous prompt vectors** prepended to the input. This is the most parameter-efficient method in the PEFT family, often achieving comparable performance to full fine-tuning on many classification tasks.

---

## 3. ⚙️ Practical Implementation & Tools

* **The PEFT Flow:**
    1.  **Select Base Model:** Choose a strong pre-trained model (e.g., LLaMA, BERT).
    2.  **Select PEFT Method:** Choose LoRA (higher capacity) or Prompt Tuning (minimal resources).
    3.  **Inject Trainable Modules:** Insert the small trainable components (e.g., $A$ and $B$ matrices for LoRA) into the model's key layers (often the attention matrices $Q$ and $V$).
    4.  **Training:** Train **only** the injected parameters on the task-specific dataset. The large base model weights remain frozen.
    5.  **Inference:** During deployment, the small adapter matrices are added back to the frozen weights (LoRA) or prepended to the input (Prompt Tuning) for the final specialized inference.
* **Tools:**
    * **Hugging Face PEFT Library:** Provides easy-to-use implementations of LoRA, Prefix Tuning, Prompt Tuning, and other PEFT methods.
    * **DeepSpeed/Accelerate:** Frameworks used to manage memory and distributed training for even more efficient parameter updates.

---

## 4. ⚖️ Trade-offs and Real-World Impact

* **Trade-off (Prompt Tuning vs. LoRA):**
    * **Prompt Tuning:** $\uparrow$ Efficiency (minimal memory/storage), $\downarrow$ Expressivity/Performance (better for simple tasks).
    * **LoRA:** $\uparrow$ Expressivity/Performance (better for complex tasks), $\downarrow$ Efficiency (more parameters than Prompt Tuning, but vastly better than FFT).
* **Impact:** PEFT methods have **democratized LLM customization**. They make it possible for smaller companies or even individual researchers to fine-tune state-of-the-art models for proprietary data without access to massive GPU clusters, leading to faster development cycles and reduced cloud computing costs.
* **The Critical Bottleneck:** In LLM fine-tuning, the bottlenecks are typically **VRAM usage** (for storing full model weights and optimizer states) and **I/O bandwidth**. PEFT minimizes the number of parameters and optimizer states that need to be updated and stored, directly addressing these bottlenecks.

---

## 5. 🧪 Hands-on Lab Preview

* **What you will do:** Use the PEFT library to apply **LoRA** to a pre-trained LLM for a text classification task. You will measure and compare the **number of trainable parameters** and the **required VRAM/GPU memory** for LoRA versus the full fine-tuning baseline.
* **Key Skill Acquired:** Quantifying the efficiency gains of PEFT and learning to implement the most practical and state-of-the-art methods for customizing large generative models.

***


## References

- EfficientML.ai Course | 2023 Fall | MIT 6.5940: [ Complete course video series ](https://youtube.com/playlist?list=PL80kAHvQbh-pT4lCkDT53zT8DKmhE0idB&si=Uu00N0zKopEixhw3).
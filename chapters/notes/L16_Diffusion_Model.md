# Lecture 16: Diffusion Model Efficiency

## Quick Reference

|Item|Reference|
|---|---|
| Slides | [View Slides](https://drive.google.com/drive/folders/1A3P6IBuS8wCzLlpdRiQBO9b1uoK3pnPf?usp=sharing)|
| Video | [EfficientML.ai Lecture 16](http://www.youtube.com/watch?v=6cAmS-_vEh8)  |
|Lab| -- |
|Professor|[Song Han](https://github.com/songhan)|

## 1. 🎯 Why It Matters for Edge AI

* **The Core Problem:** Generative AI, particularly **Diffusion Models** (like Stable Diffusion, Midjourney, DALL-E), achieves incredible results but is extremely **slow and computationally expensive**. Generating a single high-resolution image requires hundreds of sequential **denoising steps**, making real-time or low-power generation on the edge impossible with naive implementation.
* **Edge AI Benefits:** Efficient techniques are vital to bringing generative capabilities (e.g., personalized image filters, background removal, small-scale image editing) from the cloud to user devices, enabling **fast, private, and offline** generation.

---

## 2. 📝 Key Concepts and Theory

* **Diffusion Model Recap:** These models work by iteratively reversing a noise process.
    1.  **Forward (Noisy) Process:** Gradually adds Gaussian noise to an image until only pure noise remains.
    2.  **Reverse (Denoising) Process:** A neural network (**U-Net**) is trained to predict and remove the noise in each step, transforming pure noise back into a clean image.
* **The Efficiency Bottleneck: The U-Net and Sequential Steps:**
    * **The U-Net:** The backbone is typically a large U-Net, which uses numerous **Convolutional Layers** and **Self-Attention Blocks** (similar to ViTs/LLMs), consuming huge amounts of memory and FLOPs.
    * **The Sequential Process:** The core issue is the high number of **inference steps ($T$)** required for quality. Latency is $T \times \text{Latency}_{\text{U-Net}}$.
* **Key Efficiency Strategy: Denoising Speedup:**
    * **Sampling Acceleration:** Using advanced numerical solvers (e.g., **DDIM, DPM-Solver**) that allow the model to skip many steps, reducing $T$ from hundreds to just 20-50, which is the most effective speedup technique.
    * **Knowledge Distillation for Sampling:** Training a student U-Net to match the output of the teacher U-Net in fewer steps. For example, distilling a 100-step teacher into a 10-step student.

---

## 3. ⚙️ Practical Implementation & Tools

* **Architectural Optimization:**
    * **Pruning & Quantization:** Applying techniques from Lectures 3-6 to the U-Net's weights. The U-Net's **Convolutional Layers** are highly amenable to traditional pruning and quantization.
    * **Attention Optimization:** Using optimized attention mechanisms (like **FlashAttention**) on the U-Net's internal Transformer blocks to reduce memory bandwidth and latency (similar to LLMs/ViTs).
* **System-Level Optimization:**
    * **Model Partitioning:** Splitting the generative task (e.g., prompt embedding and denoising) across heterogeneous hardware (e.g., using a high-end CPU for prompt tokenization and a dedicated NPU/GPU for the U-Net inference).
* **Latent Space Reduction:** Using a smaller, compressed **Latent Space** (as in Latent Diffusion Models) ensures that the massive U-Net operates on low-resolution feature maps instead of high-resolution pixel data, saving vast amounts of computation.

---

## 4. ⚖️ Trade-offs and Real-World Impact

* **Trade-off: Speed vs. Quality:** The primary trade-off in Diffusion Models. Reducing the number of sampling steps ($T$) too aggressively leads to noticeable artifacts and lower image fidelity. Efficient methods aim to minimize $T$ without sacrificing quality.
* **Memory Footprint:** The memory consumption of the U-Net is dominated by the **Intermediate Activation Tensors**. Techniques like **Gradient Checkpointing** (used in training) are sometimes adapted for inference to trade compute time for massive memory savings.
* **Deployment Status:** While still compute-intensive, optimized diffusion models are now runnable on mobile GPUs and high-end laptops (e.g., Apple Silicon), moving them into the domain of high-end consumer Edge AI.

---

## 5. 🧪 Hands-on Lab Preview

* **What you will do:** Work with a small, pre-trained latent diffusion model. You will implement and compare the image generation time and quality using two different samplers (a slow, basic solver vs. an optimized DDIM/DPM-Solver), quantifying the massive speedup gained from **sampling acceleration**.
* **Key Skill Acquired:** Understanding the unique computational profile of generative models and applying optimization techniques to the sequential steps of the inference process.


## References

- EfficientML.ai Course | 2023 Fall | MIT 6.5940: [ Complete course video series ](https://youtube.com/playlist?list=PL80kAHvQbh-pT4lCkDT53zT8DKmhE0idB&si=Uu00N0zKopEixhw3).
# **Lesson 18: Diffusion Models**

provides a comprehensive overview of deep generative learning, focusing on the mechanics of Denoising Diffusion Probabilistic Models (DDPM), efficient variants like Latent Diffusion, and advanced techniques for acceleration and personalization.

### **I. Diffusion Model Basics (DDPM)**
Diffusion models are generative models that learn to create data by reversing a noise-adding process.
*   **Forward Process (Fixed):** This process "destroys" data by gradually adding small amounts of Gaussian noise until the image becomes pure noise ($x_T \approx \mathcal{N}(0, I)$).
*   **Reverse Process (Generative):** The model learns a denoising function to "create" data by iteratively removing noise from a starting point of stationary noise.
*   **Training & Sampling:** During training, a neural network is optimized using **MSE loss** to predict the specific noise $\epsilon$ added to an image at a given timestep $t$. Sampling involves starting from pure noise and using the trained model to iteratively estimate the original image.

### **II. Advanced Diffusion Architectures**
*   **Latent Diffusion Models (LDM):** To improve efficiency, LDMs apply the diffusion process in a compressed **latent space** rather than pixel space. This uses a pre-trained Variational Auto-Encoder (VAE) to encode images into smaller latents, leading to faster synthesis and reduced computational requirements.
*   **Conditional Generation:** Diffusion models can be guided by class labels or text prompts using techniques like **classifier guidance**, which trades off sample diversity for higher fidelity to the condition.

### **III. Editing and Personalization**
*   **Image Editing:** Techniques like **SDEdit** enable image-to-image translation and stroke-based editing. By perturbing a user's stroke or an input image with noise and then running the reverse diffusion process, the model generates a realistic output that follows the input's structure.
*   **Personalization (DreamBooth):** This allows users to fine-tune a model on just 3–5 images of a specific subject (e.g., a specific dog or clock). By using a unique identifier, the model can then generate that specific subject in entirely new contexts.

### **IV. Fast Sampling Techniques**
The standard DDPM sampling process is slow, often requiring 1000 steps. Several methods reduce this bottleneck:
*   **DDIM (Denoising Diffusion Implicit Models):** A non-Markovian approach that shares the same training objective as DDPM but allows for **fewer steps** (e.g., 10–100) during sampling without significant quality loss.
*   **Progressive Distillation:** This method distills a teacher sampler into a student model that achieves the same result in half the steps. By repeating this process, a model can be distilled down to as few as 1–4 sampling steps.

### **V. System Acceleration Techniques**
The lecture highlights three primary methods developed by the Han Lab to accelerate diffusion:
*   **SVDQuant (Quantization):** Addresses the "outlier" problem in 4-bit quantization by using **Singular Value Decomposition** to absorb outliers into low-rank components. This achieves a **3.5x speedup** and **3.6x memory savings** on models like Flux.1-dev.
*   **SIGE (Sparsity):** The **Sparse Incremental Generative Engine** optimizes image editing by selectively updating only the modified spatial regions of an image. This results in up to an **8.2x reduction in computation** and enables interactive editing on devices like a MacBook Pro.
*   **DistriFusion (Parallelism):** A distributed parallel inference framework that splits high-resolution images across multiple GPUs. It achieves up to a **6.1x speedup using 8 GPUs** by overlapping communication and computation.
*   **SANA:** A specific model mentioned that achieves a **106x speedup** in image generation compared to baseline models through deep compression and linear attention.

## References
- Source: https://notebooklm.google.com/notebook/39b7bd5c-37ef-451c-83dc-98b62ec8dc04
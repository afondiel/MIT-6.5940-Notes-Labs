# **Lesson 14: LLM Post-Training** 

covers the essential techniques for refining Large Language Models (LLMs) after their initial pre-training. This includes fine-tuning for alignment, expanding into multi-modal capabilities, and utilizing prompt engineering to enhance performance.

### **I. LLM Fine-Tuning**
Fine-tuning aligns a pre-trained model's general knowledge with specific user interaction styles or task requirements.
*   **Supervised Fine-Tuning (SFT):** The model is trained on a curated dataset of desired responses using a next-word prediction objective. This process moves the model from "dry" responses to helpful assistant-style communication.
*   **Reinforcement Learning from Human Feedback (RLHF):** This method aligns AI with human preferences (like creativity and truthfulness) by training a reward model to score outputs, which then guides the model’s policy via reinforcement learning.
*   **Direct Preference Optimization (DPO):** A simplified alternative to RLHF that treats preference optimization as a single-phase SFT task, eliminating the need for separate reward models or complex RL algorithms.
*   **Parameter-Efficient Fine-Tuning (PEFT):** Methods to update only a fraction of parameters to reduce storage and compute costs:
    *   **BitFit:** Updates only the model's bias terms.
    *   **Adapters:** Inserts small, learnable layers within the Transformer blocks.
    *   **LoRA (Low-Rank Adaptation):** Injects parallel trainable rank decomposition matrices into layers. These can be fused with pre-trained weights to ensure **zero extra inference latency**.
    *   **QLoRA:** Combines LoRA with 4-bit quantization and paged optimizers to enable fine-tuning on consumer-grade GPUs.
    *   **BitDelta:** Quantizes the weight difference between a base and fine-tuned model to 1-bit, allowing for efficient multi-tenant model serving.

### **II. Multi-modal LLMs**
Multi-modal models, or Vision-Language Models (VLMs), enable LLMs to process visual data alongside text.
*   **Architectural Approaches:** 
    *   **Cross-Attention (Flamingo):** Injects visual information into a frozen LLM using gated cross-attention layers and a **Perceiver Resampler** to map large feature maps to a fixed number of visual tokens.
    *   **Visual Tokens as Input (PaLM-E/VILA):** Treats visual inputs as discrete tokens that are fed directly into the model alongside text tokens.
*   **VILA-U:** A unified foundation model that integrates visual understanding and generation by using a unified vision tower that captures both semantic features (via contrastive loss) and appearance features (via reconstruction loss).

### **III. Prompt Engineering**
Prompt engineering allows users to improve model reasoning and factual accuracy without updating the model's weights.
*   **In-Context Learning (ICL):** Providing task descriptions or examples within the prompt to guide model behavior.
*   **Chain-of-Thought (CoT):** Asking the model to "think step by step" to generate intermediate reasoning steps, which significantly improves performance on complex tasks.
*   **Retrieval-Augmented Generation (RAG):** Combines LLMs with an external retrieval system to fetch relevant documents from a datastore. This allows the model to access up-to-date information without requiring retraining or full-model fine-tuning.

## References
- Source: https://notebooklm.google.com/notebook/39b7bd5c-37ef-451c-83dc-98b62ec8dc04
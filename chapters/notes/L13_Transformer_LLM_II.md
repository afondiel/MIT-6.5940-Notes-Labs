# Lecture 13: Transformer and LLM (Part II)

## Quick Reference

|Item|Reference|
|---|---|
| Slides | [Slides](https://drive.google.com/drive/folders/1A3P6IBuS8wCzLlpdRiQBO9b1uoK3pnPf?usp=sharing)|
| Video | [EfficientML.ai Lecture 13 - Transformer and LLM (Part II)](https://www.youtube.com/watch?v=caEDCA3Jo_4)  |
|Lab| -- |
|Professor|[Song Han](https://github.com/songhan)|


## **💡 Efficient Inference Algorithms**

The lecture begins by focusing on methods to compress LLMs to reduce memory footprint and improve latency, primarily through **quantization** and **sparsity**.

* **Quantization**  
  * **Smooth Quant / 8-bit Quantization** \[[01:01](http://www.youtube.com/watch?v=caEDCA3Jo_4&t=61)\]: This technique addresses the issue of outliers in LLM activations that cause quantization difficulty. It smooths the activations by migrating the quantization difficulty from the activation to the weights.  
  * **AWQ (Activation-aware Weight Quantization) / 4-bit Quantization** \[[01:09](http://www.youtube.com/watch?v=caEDCA3Jo_4&t=69)\]: For scenarios with small batch sizes, this method focuses on weight-only quantization to save memory bandwidth. It determines important weights by looking at the magnitude of the **activations**, not the weights themselves, and uses scaling factors to avoid mixed-precision inference.  
  * **Tiny Chat** \[[07:17](http://www.youtube.com/watch?v=caEDCA3Jo_4&t=437)\]: A practical application demonstrating the deployment of 4-bit compressed LLMs (using AWQ) on edge devices like laptops and mobile GPUs for fast, local inference.  
* **Sparsity and Pruning**  
  * **Weight Pruning** \[[32:05](http://www.youtube.com/watch?v=caEDCA3Jo_4&t=1925)\]: Introduces the concept that pruning criteria should be based on **weight times activation** magnitude, rather than just the weight's magnitude, to protect weights corresponding to large activations.  
  * **Activation Sparsity (Token Pruning)** \[[33:27](http://www.youtube.com/watch?v=caEDCA3Jo_4&t=2007)\]: Discusses methods to prune tokens in the attention mechanism based on their cumulative attention scores, leading to Cascade Token Pruning.

---

## **⚙️ Efficient Systems and Architectures**

This section focuses on system-level solutions to manage the overhead of LLM serving.

* **vLLM (Virtual Large Language Model) / PageAttention** \[[41:43](http://www.youtube.com/watch?v=caEDCA3Jo_4&t=2503)\]: Analogous to virtual memory in operating systems, this technique uses a block table to map the logical KV cache to non-contiguous physical memory blocks. This dynamic allocation eliminates external fragmentation and significantly reduces wasted memory when serving multiple users with variable-length sequences.  
  * It also allows for **KV cache sharing** \[[50:02](http://www.youtube.com/watch?v=caEDCA3Jo_4&t=3002)\] across different sequences for parallel sampling, such as generating multiple suggestions for a coding prompt.  
* **Streaming LLM** \[[01:00:54](http://www.youtube.com/watch?v=caEDCA3Jo_4&t=3654)\]: Addresses the challenge of long, continuous interactions (like multi-round dialogues) that can lead to out-of-memory errors and performance degradation.  
  * It introduces the **Attention Sync** phenomenon \[[52:55](http://www.youtube.com/watch?v=caEDCA3Jo_4&t=3175)\], noting that the first few tokens are disproportionately important. The solution is to **pin** the first few pages (attention syncs) in the KV cache and use a rolling window for the remaining tokens, maintaining constant memory and low perplexity for sequences up to millions of tokens.  
* **FlashAttention** \[[01:05:06](http://www.youtube.com/watch?v=caEDCA3Jo_4&t=3906)\]: Improves the speed of the attention mechanism by fusing the kernel operations (QKT, SoftMax, V) to avoid materializing the large N x N attention matrix in the slow main memory (DRAM), leveraging fast SRAM.  
* **Speculative Decoding** \[[01:06:36](http://www.youtube.com/watch?v=caEDCA3Jo_4&t=3996)\]: Speeds up auto-regressive decoding by using a small, fast "draft" model to generate several tokens, which are then verified simultaneously in a **batched manner** by the large, high-quality model, converting sequential decoding into a more parallel operation.

---

## **🛠️ Efficient Fine-Tuning Techniques**

The final part of the lecture covers parameter-efficient methods for customizing LLMs.

* **LoRA (Low-Rank Adaptation)** \[[01:09:34](http://www.youtube.com/watch?v=caEDCA3Jo_4&t=4174)\]: Freezes the pre-trained weights and introduces a small, low-rank bypass branch (A and B matrices) that is trained instead of the full model, drastically reducing the number of trainable parameters.  
  * **QLoRA** \[[01:11:11](http://www.youtube.com/watch?v=caEDCA3Jo_4&t=4271)\]: Combines LoRA with 4-bit quantization, using a 4-bit base model for inference while training the low-rank branch using 16-bit precision.  
* **Adapter** \[[01:12:11](http://www.youtube.com/watch?v=caEDCA3Jo_4&t=4331)\]: Similar to LoRA but inserts small, tunable layers into the main branch *in series* rather than *in parallel*.  
* **Prompt Tuning** \[[01:12:53](http://www.youtube.com/watch?v=caEDCA3Jo_4&t=4373)\]: Keeps the entire large language model fixed and only tunes a set of soft, learnable prompt vectors prepended to the input. This allows a single, fixed model to be used for multiple tasks by simply changing the prompt, which is efficient for batch inference.

---

## **🌐 Industry Application**

The lecture concludes by highlighting how these techniques are integrated into state-of-the-art serving infrastructures, such as the **NVIDIA TensorRT-LLM** \[[01:15:23](http://www.youtube.com/watch?v=caEDCA3Jo_4&t=4523)\], which uses Group-Query Attention (GQA), in-flight batching, PageAttention, Smooth Quant, and other methods covered in the course.
## References

- EfficientML.ai Course | 2023 Fall | MIT 6.5940: [ Complete course video series ](https://youtube.com/playlist?list=PL80kAHvQbh-pT4lCkDT53zT8DKmhE0idB&si=Uu00N0zKopEixhw3).
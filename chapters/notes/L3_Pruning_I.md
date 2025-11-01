# Lecture 3: Pruning and Sparsity (Part I)

## Quick Reference

|Item|Reference|
|---|---|
| Slides | [View Slides](https://drive.google.com/drive/folders/1A3P6IBuS8wCzLlpdRiQBO9b1uoK3pnPf?usp=sharing)|
| Video | [EfficientML.ai Lecture 3 - Pruning and Sparsity (Part I)](https://www.youtube.com/watch?v=95JFZPoHbgQ)  |
|Lab| [Lab1.ipynb](../../lab/notebooks/Lab1.ipynb) |
|Professor|Song Han|


## **1\. Introduction to Pruning**

**Pruning** is a model compression technique that removes redundant connections (weights) from a pre-trained network, resulting in a **sparse**, **smaller model**.

### **A. The Core Idea**

- **Observation:** Large, over-parameterized neural networks contain significant redundancy; many weights are close to zero and contribute little to the final prediction.  
- **Process:** Identify these unimportant connections and set their weights to zero, effectively removing them.  
- **Benefit:** Reduces model size (storage) and potentially computation (FLOPs/MACs).

### **B. Pruning Pipeline**

1. **Train** a dense, large network.  
2. **Prune:** Identify and zero out connections based on a chosen **criterion**.  
3. **Fine-tune (Retrain):** Re-train the remaining sparse network to recover accuracy lost during pruning.  
4. **Iterate:** Repeat the prune and fine-tune steps until the desired sparsity is reached (Iterative Pruning).

## **2\. Pruning Criteria and Granularity**

### **A. Criteria: What to Prune?**

The most common and effective criterion is based on **weight magnitude**.

- **Magnitude Pruning:** Prune the weights with the smallest absolute values ($|W|$).  
  * **Assumption:** Small weights contribute least to the output and are least essential.  
- **Other Criteria:** Hessian-based (2nd-order derivatives), Taylor expansion, etc. (More accurate but computationally expensive).

### **B. Granularity: What is the Structure of Sparsity?**

The type of structure imposed by pruning determines the potential hardware speedup.

1. **Unstructured Pruning (Fine-Grained):**  
   * **Mechanism:** Prunes individual, arbitrary weights anywhere in the network.  
   * **Compression:** Achieves the **highest compression ratio** for a given accuracy.  
   * **Hardware Challenge:** The resulting sparsity is irregular, making it difficult for standard dense hardware (CPUs/GPUs) to skip the zero-MAC operations, resulting in **zero speedup**. Requires specialized sparse hardware (L4 topic).

2. **Structured Pruning (Coarse-Grained):**  
   * **Mechanism:** Removes entire groups of weights, such as filters (channels) or rows/columns.  
   * **Compression:** Lower compression ratio than unstructured pruning.  
   * **Hardware Benefit:** The resulting architecture is a smaller, **dense** network. Standard hardware benefits immediately because the dense matrix size is simply smaller, requiring no complex indexing.  
     * **Filter Pruning:** Removes an entire output channel (filter), reducing $C\_{out}$.  
     * **Neuron Pruning:** Removes an entire neuron, zeroing out an entire row/column in the weight matrix.

## **3\. Pruning: The Reality of Speedup**

The relationship between parameter/FLOP reduction and actual speedup is complex.

- **Pruning is Lossless Compression (Storage):** A pruned network (e.g., 90% sparse) requires only 10% of the memory to store weights (using sparse formats like CSR). This is a **guaranteed 10x reduction in model size.**  
- **Pruning is *NOT* Guaranteed Speedup (Inference):**  
  * **Unstructured:** Zero speedup on standard dense hardware due to **Indirection Overhead** (the cost of looking up sparse indices).  
  * **Structured:** Guaranteed speedup because the operation becomes smaller, dense matrix multiplication.

### **A. The Iterative Approach**

- **Iterative Pruning:** Pruning in one shot is difficult to recover from. Instead, the process is repeated: prune a small fraction (e.g., 10%), fine-tune, prune the next 10%, fine-tune, and so on. This allows the network to adapt gradually.  
- **The Importance of Fine-Tuning:** The fine-tuning step is crucial for **accuracy recovery**. It compensates for the information loss by slightly adjusting the remaining, non-zero weights.

## **4\. Practical Implementation: Deep Compression**

**Deep Compression** is an optimization framework that combines pruning, quantization (L5), and Huffman coding (L18) into a three-stage pipeline to achieve maximal compression.

- **Pruning:** Reduces redundant connections.  
- **Quantization:** Clusters weights and shares values (reducing bitwidth).  
- **Huffman Coding:** Further losslessly compresses the resulting sparse, quantized indices.

## References

- EfficientML.ai Course | 2023 Fall | MIT 6.5940: [ Complete course video series ](https://youtube.com/playlist?list=PL80kAHvQbh-pT4lCkDT53zT8DKmhE0idB&si=Uu00N0zKopEixhw3).
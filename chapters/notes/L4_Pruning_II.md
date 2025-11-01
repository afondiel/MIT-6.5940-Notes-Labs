# Lecture 4: Pruning and Sparsity (Part II)

## Quick Reference

|Item|Reference|
|---|---|
|Slides | [View Slides](https://drive.google.com/drive/folders/1A3P6IBuS8wCzLlpdRiQBO9b1uoK3pnPf?usp=sharing)|
|Video | [EfficientML.ai Lecture 4 - Pruning and Sparsity (Part II)](https://www.youtube.com/watch?v=sDJymyfAOKY)  |
|Lab| [Lab1.ipynb](../../lab/notebooks/Lab1.ipynb) |
|Professor|Song Han|


## **1\. Pruning Criteria: First-Order Information**

While **Weight Magnitude Pruning** (L3) is the simplest method, and **Second-Order Information (Hessian)** is too expensive, **First-Order Taylor Expansion** provides a computationally cheap and effective alternative.

### **A. Taylor Pruning (First-Order Saliency)**

- **Concept:**
  Measures the estimated change in the loss function ($\Delta L$) if a specific weight ($w{i}$) is removed (set to zero).  
- **Approximation:** The change in loss is approximated using the first-order Taylor series expansion:

$$
\Delta L = L(w{i}=0) - L(w{i}) \approx - 
\frac{\partial L}{\partial w{i}} \cdot w{i}
$$  

- **Taylor Magnitude (Saliency Score)**:  
  The absolute value of this loss change is used as the importance score.
  $$
  \text{Taylor Magnitude} = \left| \frac{\partial L}{\partial w{i}} \cdot w{i} \right|$$

- **Pruning Rule:** Weights with the **smallest Taylor Magnitude** are considered least important and should be pruned.  
- **Advantage over** $|w{i}|$ 
  By incorporating the gradient ($\frac{\partial L}{\partial w{i}}$), Taylor Pruning provides a more accurate estimate of importance. A small weight with a large gradient is still critical, and this method captures that importance.

## **2\. Efficient Storage of Unstructured Sparsity**

Unstructured pruning yields high compression but results in sparse, irregular matrices. Storing all the zeros wastes memory.

- **Solution: Compressed Sparse Formats**
  * To save memory, specialized data structures like **Compressed Sparse Row (CSR)** or **Compressed Sparse Column (CSC)** are used to store only the non-zero values, their indices, and pointers.  
- **Hardware Challenge:** 
  The irregular structure and complex indexing require specialized hardware, such as the **Efficient Inference Engine (EIE)**, to efficiently skip zero computations and manage addressing.

## **3\. When to Prune: The Lottery Ticket Hypothesis (LTH)**

The LTH challenges the conventional iterative pruning pipeline by asking which component is truly essential for high accuracy: the *final weights* or the *initialization*.

### **A. The Hypothesis**

-  **Statement:** A dense, randomly initialized neural network contains a sparse **subnetwork** (the "winning ticket") that, when trained in isolation (starting from the *original* initial weights), can achieve comparable accuracy to the full, dense network.

### **B. LTH Pipeline vs. Traditional IMP**

### B. LTH Pipeline vs. Traditional IMP

| Pipeline        | Train         | Prune                               | Reversion                               | Final Step                 |
|-----------------|---------------|-----------------------------------|----------------------------------------|----------------------------|
| Traditional IMP | Full training | Remove low magnitude weights $(\left\|w \right\| < \tau)$ | Reinitialize remaining weights          | Retrain pruned network      |
| LTH             | Full training | Remove low magnitude weights $(\left\|w \right\| < \tau)$ | Reset remaining weights to initial values | Retrain pruned network      |

- **Key Finding:**
  The crucial factor is finding the **subnetwork structure** and retaining the **original random initialization** for that structure.

## **4\. Dynamic Pruning and Hardware-Aware Sparsity**

### **A. Dynamical Sparse Training (DST)**

- **Goal:** Avoid the costly initial full training required by LTH by finding the effective subnetwork early.  
- **Mechanism:** DST allows the network's sparse connectivity pattern to **change dynamically** during training. When a weight is pruned, a new, potentially more critical connection can be "grown" in a different location.  
- **Benefit:** 
  Achieves better sparsity and accuracy by exploring the connectivity space while skipping the initial dense model training.

### **B. Hardware-Aware Sparsity**  

To bridge the gap between high sparsity (unstructured) and hardware efficiency (structured), hybrid methods are used.

- **Block-Wise Sparse Structure:** Instead of single weights, we prune a small, regular **block of weights** (e.g., $2 \times 2$ or $4 \times 4$).  
- **Example: NVIDIA 2:4 Sparsity:** 
  Modern hardware like the NVIDIA A100 GPU is specifically designed to accelerate a fixed sparse structure: in every block of four weights, at least two must be zero. This co-design between model compression and hardware provides guaranteed speedup.

## **Summary of Pruning Trade-offs**


- **Unstructured:** Max **Sparsity** $\rightarrow$ Requires **Specialized Hardware** (EIE).  
- **Structured:** Guaranteed **Speedup** on standard hardware $\rightarrow$ Lower achievable sparsity.  
- **Block-Wise:** Best **Compromise** $\rightarrow$ High sparsity with **Regularity** for hardware acceleration.

## References

- EfficientML.ai Course | 2023 Fall | MIT 6.5940: [ Complete course video series ](https://youtube.com/playlist?list=PL80kAHvQbh-pT4lCkDT53zT8DKmhE0idB&si=Uu00N0zKopEixhw3).
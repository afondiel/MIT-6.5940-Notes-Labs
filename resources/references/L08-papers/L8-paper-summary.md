# 🤖 Lecture 8: Neural Architecture Search (NAS) & System Efficiency

## 11. NAS with Reinforcement Learning (2017)

* **Problem**: Designing high-performing network architectures requires expert, trial-and-error labor.
* **Solution**: Trained an RNN controller via RL policy gradients to generate strings.
* **Findings/Results**: Found exceptional topologies competing directly with elite human designs.
* **Challenge**: Disastrously expensive compute demands requiring thousands of GPU hours. 

## 12. NASNet (2018)

* **Problem**: Searching over massive global network topologies does not scale to large images.
* **Solution**: Searched for modular Normal/Reduction cells instead of whole networks.
* **Findings/Results**: Cells easily transferred across diverse target vision datasets.
* **Challenge**: Structural cell complexity hurts raw on-device execution speed. 

## 13. DARTS: Differentiable Architecture Search (2019)

* **Problem**: Discrete search methods (RL/Evolution) are slow due to non-differentiable evaluation.
* **Solution**: Formulated a continuous relaxation of the architectural selection space.
* **Findings/Results**: Enabled network optimization using fast, standard gradient descent.
* **Challenge**: High memory consumption due to tracking all candidate operations concurrently. 

## 14. ProxylessNAS (2019)
 
* **Problem**: Proxy tasks (like training on proxy datasets) yield suboptimal target-hardware performance.
* **Solution**: Searched directly on the target hardware without proxies using path binarization.
* **Findings/Results**: Drastically reduced memory overhead to match standard training footprints.
* **Challenge**: Complex optimization landscape with mixed architectural and weight parameter steps.

## 15. FBNet (2019)

* **Problem**: Floating-point operation (FLOP) counts correlate poorly with real device latency.
* **Solution**: Integrated real-world device latency look-up tables into differentiable search.
* **Findings/Results**: Discovered highly hardware-optimized models tailored to specific mobile chips.
* **Challenge**: Requires building distinct lookup tables for every single device target. 

## 16. Single Path One-Shot NAS (2020)

* **Problem**: Weight sharing methods create severe weight co-adaptation, skewing performance rankings.
* **Solution**: Decoupled the supernet training phase entirely from the architecture search.
* **Findings/Results**: Standardized supernet paths via uniform random sampling for fair rankings.
* **Challenge**: Training an all-encompassing supernet requires massive initial search investments. 

## 17. Once-for-All (OFA) (2020)

* **Problem**: Running separate NAS routines for thousands of distinct hardware profiles is impossible.
* **Solution**: Trained a single master supernet supporting diverse sub-networks natively.
* **Findings/Results**: Cut cumulative search costs down to zero for subsequent target devices.
* **Challenge**: High engineering complexity needed to prevent progressive shrinking interference.

## 18. MCUNet (2020)
 
* **Problem**: Microcontrollers possess tiny memory spaces, blocking standard deep learning deployment.
* **Solution**: Jointly co-designed the efficient search space alongside a tiny compiler runtime.
* **Findings/Results**: Enabled ImageNet-scale deep learning models inside tight sub-megabyte SRAM budgets.
* **Challenge**: Demands custom tailored optimization layers for every new microcontroller target. 


## References
- https://github.com/afondiel/MIT-6.5940-Notes-Labs/blob/main/resources/references/L08-papers/L8-paper-references.md


## Notebook Summary

This notebook implements **Neural Architecture Search** on a **Once-for-All** super network (*Cai et al., 2020*), searching for subnets that hit hard edge constraints on the **Visual Wake Words** dataset. The point of the lab is that NAS on a pretrained super network is a *search* problem, not a training problem: every candidate is evaluated by predictors in milliseconds instead of being trained.

### Part 0: Super Network and the Design Space

The OFA super network exposes four search dimensions: width multiplier (`wid`), per-block kernel size (`ks` ∈ {3, 5, 7}), per-block expand ratio (`e` ∈ {3, 4, 6}), depth per stage (`d`), and input `image_size`.

*   **Sampled subnets** span a wide range — 0.7M params → **86.6%** accuracy, 1.5M params → **87.9%**.
*   **Question 1 — which dimension matters most:** width multiplier and input resolution dominate accuracy; kernel size and expand ratio matter less.
*   **Design space bounds** (cell 39): smallest subnet **8.3M MACs / 72.0 KB** peak memory, largest **79.4M MACs / 270.0 KB**.

### Part 1: Predictors

Search is only cheap if scoring a candidate is cheap. Two predictors replace measurement and training:

*   **Efficiency predictor (Question 2)** — computes MACs and peak memory analytically from a subnet config, no forward pass.
*   **Accuracy predictor (Questions 3–4)** — a 4-layer MLP (128 → 400 → 400 → 400 → 1) over a one-hot encoding of kernel size and expand ratio per block, trained on a dataset of pre-evaluated subnets.
*   **Calibration check:** mean accuracy across the dataset is **90.3%**; the predictor's chosen subnet scores **91.0%** on the holdout validation set, and predicted-vs-real accuracy correlates tightly (cell 52).

### Part 2: Search

**Random search (Question 5)** vs **evolutionary search (Questions 6–8)** under matched constraints:

| Constraint | Random search | Evolutionary search |
|---|---:|---:|
| MACs ≤ 50M | **93.28%** | 92.26% |
| MACs ≤ 100M | **93.45%** | 92.83% |
| Peak memory ≤ 256KB | **92.83%** | 92.26% |
| Peak memory ≤ 512KB | 93.15% | **93.28%** |

Random search is competitive here and wins three of four — a useful negative result. Evolutionary search is sensitive to `evo_params` (population size, mutation rate, parent ratio), and the default settings are not tuned per constraint.

### Part 2b: Real-World Constraints (Question 9)

Visual Wake Words targets MCU-class deployment, so both MACs *and* peak memory bind simultaneously. Evolutionary search with per-task tuned `evo_params`:

| Constraint | Accuracy | Credit threshold | Result |
|---|---:|---|---|
| 60M MACs **and** 250 KB | **92.93%** | ≥ 92.5% | full credit |
| 30M MACs **and** 200 KB | **90.15%** | ≥ 90% (bonus) | bonus earned |

### Question 10: Design Space Limits

*   **A — activation ≤ 256KB and MACs ≤ 15M: Yes.** The smallest subnet in the space sits at **72.0 KB / 8.3M MACs** (cell 70), comfortably inside both bounds.
*   **B — activation ≤ 64 KB: No.** 72.0 KB is the floor of this design space; no subnet can go below it without changing the super network itself.

### Key Takeaways

1.  **Predictors are what make NAS tractable.** Analytic efficiency plus a learned accuracy predictor turn each candidate evaluation into a lookup, which is what allows thousands of subnets to be scored per search.
2.  **Evolutionary search is not automatically better than random.** At these budgets random search matched or beat it in 3 of 4 settings. The advantage of evolution shows up once constraints get tight and multi-dimensional — which is exactly where Question 9 lives.
3.  **Peak memory, not MACs, is the binding constraint on MCUs.** The design space floor is 72 KB of activation, and that floor — not compute — is what rules out Question 10B.
4.  **The search space sets the ceiling and the floor.** No search strategy can produce a 64 KB subnet from a super network whose minimum is 72 KB.

## References
- Lab3: https://github.com/afondiel/MIT-6.5940-Notes-Labs/tree/main/lab/notebooks/lab3
- **Once-for-All: Train One Network and Specialize it for Efficient Deployment** (*Cai et al., 2020*) — https://arxiv.org/abs/1908.09791
- **MCUNet: Tiny Deep Learning on IoT Devices** (*Lin et al., 2020*) — https://arxiv.org/abs/2007.10319
- Lecture notes: [L07 NAS I](../../../chapters/notes/L07_NAS_I.md), [L10 MCUNet](../../../chapters/notes/L10_MCUNet.md)

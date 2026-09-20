# **Notebook Summary**

This notebook implements **AWQ** (*Lin et al., 2023*) — activation-aware weight-only quantization — on `facebook/opt-1.3b`. The goal is W3A16: compress weights to 3 bits while leaving activations in FP16, then recover the perplexity that naive quantization destroys. Perplexity is measured on **wikitext-2**; the activation statistics that drive the method come from the **Pile** validation split.

The arc of the lab is a single argument: *not all weights matter equally, and the ones that matter are identified by activations, not by weight magnitude.*

### Section 1: Setup and Baseline

*   **Environment**: Installs `transformers`, `accelerate`, `datasets`, `sentencepiece` and `zstandard`. The model is loaded with `device_map="auto"`, so `accelerate` is required.
*   **`evaluate`**: Computes perplexity over 40 sequences of 2048 tokens from wikitext-2 test, accumulating negative log-likelihood and exponentiating the mean.
*   **`get_model_size`**: Counts parameters analytically and multiplies by a supplied bit width. With `group_size=128` it adds `(16 + 4) / 128` bits per weight to account for the per-group FP16 scale and 4-bit zero point — the storage overhead that group-wise quantization actually costs.
*   **Baseline**: `facebook/opt-1.3b` gives **perplexity 14.47** at a reported **5043.73 MiB**. Note that this size is *analytic* — `numel() × 32.16 bits`, i.e. what the model would cost in FP32 — not the memory actually occupied. See Environment Notes.

### Section 2: Naive 3-bit Group Quantization

*   **`pseudo_quantize_tensor`**: Asymmetric uniform quantization in groups of 128 — compute per-group min/max, derive scale and zero point, round and clamp into `[0, 2^b - 1]`, then **dequantize back to float**. Nothing is stored in 3 bits; the numerical effect of doing so is simulated.
*   **Result**: Size drops to **495.06 MiB** (**10.2× smaller**), but perplexity explodes to **123.88** — roughly 8.6× worse than baseline. The model is compressed and useless.

### Section 3: Protecting Salient Weights (Question 1)

*   **Calibration**: `get_calib_feat` registers a forward hook on every `nn.Linear` and records the per-channel **mean absolute activation** across 127 blocks of 512 tokens drawn from `mit-han-lab/pile-val-backup`. Channel importance is the sum of these statistics.
*   **Question 1.1 — keep the top 1% in FP16**: Preserving the 1% of input channels with the highest activation importance recovers perplexity to **17.16** (target 17.15) at unchanged size.
*   **Question 1.2 — the ablation that makes the point**: Protecting a *random* 1% instead yields **101.30** (target: over 100). Keeping 1% of channels in FP16 is worth almost nothing; keeping *the right* 1% is worth nearly everything.
*   **Question 1.3 — why**: Salient weight channels are those multiplied by **activation outliers**. In LLMs, outliers appear in a small fraction of channels and persist across all tokens, so quantization error in those channels is amplified by consistently large inputs.

### Section 4: Scaling Instead of Mixed Precision (Question 2)

Mixed precision works but is a deployment problem — an FP16 island inside an INT3 tensor complicates every kernel. AWQ's insight is that scaling achieves the same protection in uniform precision.

*   **The mechanism**: For `y = wx`, quantization error is `Δ · RoundErr(w/Δ) · x`. Scaling a channel up by `s` before quantization and dividing by `s` after leaves the output unchanged but divides its error by `s`. Because `RoundErr ≈ 0.25` regardless, and because one channel scaled up inside a group of 128 usually does not move the group maximum, `Δ` stays fixed — so the error genuinely shrinks.
*   **Question 2.1 — fixed scale**: Scaling the salient 1% by `s = 2` gives **18.95** (target 18.93) with **no mixed precision at all**.
*   **Question 2.2 — the scale sweep**: perplexity is U-shaped in `s`.

    | `scale_factor` | 1 | 2 | 3 | 4 |
    |---|---|---|---|---|
    | **Perplexity** | 123.88 | **18.95** | 19.23 | 21.24 |

    Past `s = 2` the benefit reverses: scaling a channel far enough *does* raise the group maximum, which increases `Δ` for all 128 channels in that group. Protecting one channel starts damaging its neighbours.
*   **Question 2.3 — search the scale instead of guessing it**: Rather than a single global `s`, grid-search `α` over 20 values with `s = s_x^α`, minimizing the per-block reconstruction loss `‖Q(W·s)(s⁻¹X) − WX‖`. Activation statistics alone suffice — no gradients, no fine-tuning. Result: **17.94** (target 17.92).
*   **Why it is free at inference**: `scale_ln_fcs` and `scale_fc_fc` fold `s⁻¹` into the **preceding** LayerNorm or linear layer's weights. The scaling never appears as a runtime operation — it is absorbed into weights that were already being loaded.

### Results

All figures from the completed run, `facebook/opt-1.3b`, 3-bit, group size 128:

| Stage | Perplexity | Size (MiB) | vs. FP32 | Lab target |
|---|---:|---:|---|---:|
| Baseline (size quoted FP32-equivalent) | **14.47** | 5043.73 | — | — |
| Naive 3-bit group quant | 123.88 | 495.06 | 10.2× smaller, unusable | — |
| + top 1% salient kept FP16 (Q1.1) | 17.16 | 495.06 | +18.6% perplexity | 17.15 |
| + random 1% kept FP16 (Q1.2) | 101.30 | 495.06 | ablation — saliency is what matters | > 100 |
| + fixed scale-up `s=2` (Q2.1) | 18.95 | 495.06 | no mixed precision | 18.93 |
| + searched per-channel scales (Q2.3) | **17.94** | 495.06 | **10.2× smaller, +24.0% perplexity** | 17.92 |

Every question landed within 0.02 of its target.

### Key Takeaways

1.  **Saliency is defined by activations, not weights.** The random-1% ablation (101.30 vs 17.16) is the cleanest result in the lab: identical mixed-precision budget, catastrophically different outcome. Weight magnitude alone would not have found these channels.
2.  **Scaling substitutes for mixed precision.** Searched scales reach 17.94 versus 17.16 for the FP16-island approach — within ~5% perplexity, in a uniform 3-bit format that ordinary integer kernels can actually execute. That gap is the price of deployability, and it is small.
3.  **Group size creates a shared fate.** The U-shaped scale sweep exists because 128 channels share one `Δ`. Every per-channel protection is bounded by what it does to the rest of its group — the same tension that makes group size the central knob in weight-only quantization.
4.  **The cost is bookkeeping, not compute.** 3 bits per weight is really 3.16 bits once the per-group FP16 scale and 4-bit zero point are counted. At group size 128 that overhead is 5%; at group size 32 it would be 21%.
5.  **Calibration is cheap.** ~65k tokens of generic web text, forward passes only — no labels, no gradients, no fine-tuning. This is what makes AWQ practical as a post-training method.

### Environment Notes

The notebook originally pinned a mid-2023 stack (`transformers==4.31.0`, `tokenizers==0.13.3`, `datasets==2.14.4`). Those pins **cannot install on Colab's current Python 3.13**: `tokenizers 0.13.3` shipped wheels only through cp311, and its Rust/PyO3 0.18 backend refuses to build against Python ≥ 3.12. The install cell now uses version floors matching `requirements.txt` and lets `transformers` resolve `tokenizers` itself.

Two consequences worth remembering when re-running:

*   `datasets` ≥ 5 rejects bare dataset ids, so wikitext is loaded as **`Salesforce/wikitext`**.
*   `zstandard` must be installed explicitly — it is only a test extra of `datasets`, but `mit-han-lab/pile-val-backup` ships as `val.jsonl.zst`.
*   **The baseline is FP16, not FP32.** `transformers` 5.x loads `from_pretrained` in the *checkpoint's* dtype (verified: `torch.float16`) instead of upcasting to FP32 as the 2023 pins did. The lab's "FP32 model" label and the 5043.73 MiB figure are therefore FP32-*equivalent* accounting — `get_model_size` multiplies parameter count by a bit width it is handed and never inspects the real dtype. This does not move the results (all four question targets matched within 0.02), but the 10.2× compression figure is against hypothetical FP32 storage, not against the ~2.6 GB actually loaded.

### References

- Lab4: https://github.com/afondiel/MIT-6.5940-Notes-Labs/tree/main/lab/notebooks/lab4
- **AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration** (*Lin et al., 2023*) — https://arxiv.org/abs/2306.00978
- Lecture notes: [L06 Quantization II](../../../chapters/notes/L06_Quantization_II.md)

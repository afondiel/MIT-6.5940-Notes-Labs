## Lab Summary

**Status: not yet completed.** No results are recorded in this directory — no report, no latency measurements, no implementation patch. The sections below describe the lab and its workflow; fill in the benchmark table once the kernels are implemented.

Unlike Lab0–Lab4, **Lab 5 is not a Jupyter notebook.** It is a C++ kernel-optimization exercise built on [TinyChatEngine](https://github.com/mit-han-lab/TinyChatEngine), deploying **LLaMA2-7B-chat** on a laptop CPU. The deliverable is optimized matmul kernels plus a measured end-to-end latency improvement — this is the lab where the course's compression theory finally meets real silicon.

### Where the Code Lives

The working tree is the **`lab/notebooks/Lab1-4/Lab5/` submodule** (read-only upstream — edits there do not commit to this repo):

```
Lab1-4/Lab5/
├── transformer/          # the lab proper
│   ├── src/ops/linear.cc # ← the file you edit: IMP dispatch + kernel bodies
│   ├── Makefile          # CXXFLAGS += -DIMP=$(IMP)
│   ├── evaluate.sh       # rebuild + test every implementation in turn
│   └── tests/
└── kernels/metal/        # macOS-only Metal backend
```

`lab/notebooks/lab5/` (this directory) holds only the instruction documents, the upstream URL, and this summary.

### The Six Implementations

`IMP` is a **compile-time** define, not a runtime environment variable. `src/ops/linear.cc` dispatches on it at line ~184:

| `IMP` | Kernel | Technique |
|---:|---|---|
| 0 | `mat_mul_reference` | Naive baseline |
| 1 | `mat_mul_loop_unrolling` | Reduce loop + branch overhead |
| 2 | `mat_mul_multithreading` | Parallelize across cores |
| 3 | `mat_mul_simd_programming` | AVX2/NEON vector intrinsics |
| 4 | `mat_mul_multithreading_loop_unrolling` | Combine 1 + 2 |
| 5 | `mat_mul_all_techniques` | All of the above |

### Workflow

```bash
cd lab/notebooks/Lab1-4/Lab5/transformer

make -j IMP=5      # build one implementation (integer, compile-time)
./test_linear      # correctness against the reference kernel
./chat             # interactive LLaMA2-7B-chat demo

./evaluate.sh                  # rebuild + benchmark every implementation
./evaluate.sh simd_programming # one implementation, by name
                               # (only this script accepts names, not make)
```

Bare `make` builds both `test_linear` and `chat`. Correctness first: `./test_linear` must pass before a latency number means anything.

### Results

To be filled in after implementation. Report latency per technique against the `IMP=0` reference, on your own ISA:

| `IMP` | Technique | Latency (ms/token) | Speedup vs reference |
|---:|---|---:|---:|
| 0 | reference | — | 1.00× |
| 1 | loop unrolling | — | — |
| 2 | multithreading | — | — |
| 3 | SIMD | — | — |
| 4 | multithreading + unrolling | — | — |
| 5 | all techniques | — | — |

### Why This Lab Matters

Lab 4 quantized weights to 3–4 bits and reported a ~10× size reduction — but a compression ratio is not a speedup. Decoding is **memory-bandwidth bound**: the arithmetic intensity of a GEMV is roughly 1 FLOP per byte, orders of magnitude below what the hardware can sustain. Smaller weights help only if a kernel actually exploits the narrower loads. This lab is where that gap closes, and where the INT4 weights from Lab 4 turn into tokens per second.

### Submission Requirements (upstream)

*   **Report** — code plus the measured improvement for each starter kernel.
*   **Patch** — `git diff` of the implementation, named `{studentID}-{ISA}.patch`, where ISA is `x86` or `ARM`.

## References
- Lab5 upstream: https://github.com/yifanlu0227/LLaMA2-7B-on-laptop/tree/aaf7bf3e7f9667d4c6170b2c3ffff39b31c089e3
- Colab: https://drive.google.com/drive/folders/1MhMvxvLsyYrN-4C6eQG8Zj2JeSuyAOf0?usp=drive_link
- [TinyChatEngine](https://github.com/mit-han-lab/TinyChatEngine) · [TinyEngine](https://github.com/mit-han-lab/tinyengine) · [AWQ](https://github.com/mit-han-lab/llm-awq)
- Lecture notes: [L11 TinyEngine & Parallel Processing](../../../chapters/notes/L11_TinyEngine_Parallel.md), [L13 Transformer & LLM II](../../../chapters/notes/L13_Transformer_LLM_II.md)

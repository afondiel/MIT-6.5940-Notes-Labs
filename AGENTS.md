# AGENTS.md

This file provides guidance to coding agents when working with code in this repository.

## What This Repository Is

Personal study notes and hands-on labs for **MIT 6.5940 — TinyML and Efficient Deep Learning Computing** (Prof. Song Han, MIT HAN Lab). It is a *content* repository first and a code repository second: the primary deliverable is Markdown lecture notes, with PyTorch notebooks and C++ kernels as supporting practice material. The framing throughout is that of a practicing **Edge AI engineer** translating research into deployable technique, not a student transcribing lectures.

Most tasks here are content work — writing, correcting, or extending notes. Treat prose accuracy with the same rigor you would treat a failing test.

## Setup

```bash
# Submodules are required — two of them, and neither is optional for lab work
git submodule update --init --recursive

pip install -r requirements.txt
```

`requirements.txt` is the single source of truth for Python deps and is annotated per-lab (which package belongs to which notebook). It is pinned by floor against Python 3.12 / torch 2.13. For a CUDA build, install torch from the PyTorch index *first*, then the requirements file — the default PyPI wheel is CPU-only on some platforms.

## Running Things

**Jupyter labs** — each lab lives in its own lowercase directory:

```bash
jupyter lab lab/notebooks/lab2/Lab2.ipynb   # lab0 … lab5
```

Notebooks import shared profiling utilities from `lab/notebooks/helper.py` (parameter counts, MACs via `torchprofile`, `Byte`/`KiB`/`MiB`/`GiB` constants, ONNX export). Anything measuring model size or FLOPs should reuse those helpers rather than recomputing byte math inline.

**C++ parallel computing tutorial** (`lab/code/parallel-computing-tutorial/`, submodule) — matmul optimized six ways (loop unrolling/reordering/tiling, multithreading, SIMD, CUDA). The Makefile auto-detects CUDA and ARM vs x86:

```bash
make -j                        # produces ./benchmark
./benchmark                    # all techniques
./benchmark SIMD_programming   # one technique
```

**Lab5 transformer kernels** (`lab/notebooks/Lab1-4/Lab5/transformer/`, nested submodule) — INT4/INT8 LLM inference kernels. `IMP` is a **compile-time** define (`-DIMP=$(IMP)`), not a runtime variable; it selects the implementation by integer:

```bash
make -j IMP=5      # 0 reference · 1 loop_unrolling · 2 multithreading
                   # 3 simd_programming · 4 multithreading_loop_unrolling · 5 all_techniques
./test_linear      # unit tests
./chat             # interactive demo

./evaluate.sh                  # rebuild + test every implementation in turn
./evaluate.sh simd_programming # or just one, by name
```

`make` with no target builds both `test_linear` and `chat`. Metal kernels (`kernels/metal/`) are macOS-only.

## Architecture

### Three content axes, two lecture numberings

Content is split across three parallel trees keyed by lecture number:

| Tree | Contents |
|---|---|
| `chapters/notes/LXX_Topic.md` | The canonical hand-written notes (L01–L23, plus LA1–LA3) |
| `chapters/slides/` | Source PDFs + `slides-summary-fall-2024/LXX-slide-summary.md` |
| `resources/references/LXX-papers/` | Papers, per-lecture reference lists, and summaries |

**Critical: `LXX` does not mean the same thing in every tree.**

- `chapters/notes/` and `resources/references/` follow the **Fall 2023** syllabus.
- `chapters/slides/` (both the `LecXX-*.pdf` files and `slides-summary-fall-2024/`) and the **README tables** follow the **Fall 2024** syllabus.

The two agree through L11 and diverge from L12 onward. For example, `chapters/notes/L14_ViT_Efficiency.md` is Vision Transformers, but `chapters/slides/Lec14-LLM Post-training.pdf` and `slides-summary-fall-2024/L14-slide-summary.md` are LLM post-training. Fall 2024's Vision Transformer material is `Lec16`, while `chapters/notes/L16_Diffusion_Model.md` is diffusion.

Never pair a note with a slide deck or slide summary by number alone. **Match on topic**, confirm by opening both, and say which numbering you used when reporting. The README's tables use Fall 2024 numbering while linking to Fall 2023 note filenames, so several of its links are already mismatched — do not treat the README as authoritative for this mapping.

### Note anatomy

Every lecture note opens with an H1 `# Lecture NN: Title`, then a `## Quick Reference` table (Slides / Video / Lab / Professor), then numbered sections. Audio-extension notes replace the Professor row with a Credit row. Follow the existing shape when adding or editing.

### The audio extension (LA1–LA3)

`LA1`–`LA3` are **not official MIT course material**. They are community-designed notes extending the course's efficiency playbook to the audio modality, which the official syllabus omits. They deliberately mirror the course's pedagogy — start from the breakthrough model, scale up, compress back down — and are anchored to a defining paper each: wav2vec 2.0 (LA1), WaveNet (LA2), CLAP (LA3). LA1 parallels L12–L13, LA2 parallels L17–L18, LA3 parallels L12 + L16 (Fall 2024 numbering).

Preserve the disclaimers distinguishing these from official material; do not present them as course canon.

### Lab layout

```
lab/notebooks/labN/          # working copies — Lab0 … Lab5, with per-lab
                             # *-notebook-summary.md and code-summary.md notes
lab/notebooks/helper.py      # shared profiling / export utilities
lab/notebooks/Lab1-4/        # SUBMODULE: upstream reference solutions
                             # (yifanlu0227/MIT-6.5940) + Lab5 kernels
lab/code/                    # SUBMODULE: parallel-computing-tutorial
```

`lab/notebooks/labN/` holds the *worked* versions; `lab/notebooks/Lab1-4/` is the read-only upstream. Edit the former, not the latter — changes inside a submodule do not commit to this repo. Files suffixed `-last` or `_last` are prior iterations kept for comparison; `chapters/notes-last/` is likewise an archive, **not** canonical.

## Conventions

**Notes** — GitHub-flavored Markdown. Title case for main headings. **Bold** for key concepts on first mention. Cite the originating paper for any named algorithm, as **LoRA** (*Hu et al., 2021*). New notes: `chapters/notes/LXX_Topic_Name.md`.

**Notebooks** — every cell must run top-to-bottom without error. **Clear all outputs before committing** — no output blobs in git. Include a heading, stated goals, sanity-check cells, and a conclusion. Report benchmarks (accuracy, size, MACs, latency) against the FP32 baseline, which is the comparison the whole course is built around.

**Python** — PEP 8. Lab5 additionally enforces black/isort/pylint/mypy at line length 120 via its own `pyproject.toml` and pre-commit config.

**Commits** — conventional prefixes (`feat:`, `fix:`, `docs:`, `labs:`) as used in the existing history. Branches: `feat/add-qlora-lab`, `fix/pruning-accuracy-in-L04`.

## Current State

- **`lab/notebooks/helper.py` does not compile** — `IndentationError` at line 26, inside `calc_parameters`. Any notebook importing it fails. Fix before running labs.
- **No note covers GAN / Video / Point Cloud.** The deck exists (`chapters/slides/Lec17-Efficient-GANs-Video-PointCloud.pdf`) but no `chapters/notes/` file corresponds to it — the L15 slot holds Advanced Sparsity instead. This is the largest content gap.
- **`Lec22` deck is missing** from `chapters/slides/` (present: Lec01–Lec21, Lec23).
- **`resources/references/` mostly tracks the notes' numbering, but not perfectly** — e.g. `L15-papers/` contains diffusion material that belongs with L16. Verify per-lecture rather than assuming.
- **Transcripts**: only `chapters/transcript/l1.md` and `l14.md` exist.
- **LA1–LA3**: notes only, no accompanying notebooks.
- **Lab5**: C++ kernels only, no Jupyter wrapper.

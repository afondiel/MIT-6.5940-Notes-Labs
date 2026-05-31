# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

Personal course notes and hands-on labs for MIT 6.5940 (TinyML and Efficient Deep Learning Computing, Fall 2023). The goal is to bridge academic/research content into practical notes for an Edge AI engineer.

## Repository Structure

```
chapters/notes/                         # Primary lecture notes (LXX_Topic_Name.md + LA1–LA3 audio extension)
chapters/notes-last/                    # Draft/previous iterations — NOT canonical, treat as archive
chapters/slides/                        # PDF lecture slides
chapters/slides/slides-summary-fall-2024/  # Markdown summaries of Fall 2024 slides (23 files)
chapters/transcript/                    # Lecture transcripts (L1 and L14 only)
lab/notebooks/                          # Jupyter notebooks (Lab0–Lab4 + advanced Lab5)
lab/code/                               # C++ parallel computing tutorial (git submodule)
resources/references/                   # Papers organized by lecture (L0–L23 + LA1–LA3)
prompt.md                               # Original AI system prompt used to generate notes
```

## Running Labs

**Jupyter notebooks:**
```bash
cd lab/notebooks/
jupyter notebook Lab1.ipynb   # or Lab0–Lab4
```

**C++ parallel computing tutorial (lab/code/parallel-computing-tutorial/):**
```bash
make -j            # auto-detects CUDA, ARM vs x86
./benchmark        # run all benchmarks
./benchmark SIMD_programming   # run a specific technique
```

**Lab5 transformer kernels (lab/notebooks/Lab1-4/Lab5/transformer/):**
```bash
make               # builds for x86 AVX2 or ARM NEON
make test_linear   # compile test binary
./test_linear      # run unit tests
IMP=all_techniques ./chat   # run the optimized chat app
```

**Metal kernels (macOS only):**
```bash
cd lab/notebooks/Lab1-4/Lab5/kernels/metal/
make
```

## Lecture Notes Conventions

- Core lectures named `chapters/notes/LXX_Topic_Name.md` (L01–L23)
- Audio extension lectures named `chapters/notes/LAX_Topic_Name.md` (LA1–LA3) — community-designed, not official
- Each note includes: quick reference table (slides/video/lab/prof), numbered sections, comparison tables, paper citations
- Key concepts in **bold** on first mention
- Citations format: **LoRA** (*Hu et al., 2021*)
- `chapters/notes-last/` holds draft/previous iterations — not canonical, treat as archive

## Lab Notebook Conventions

- All cells must run top-to-bottom without errors
- Clear all outputs before committing (no large output blobs in git)
- Include heading, goals, sanity-check cells, and conclusion
- New labs go in `lab/notebooks/` named `LXX_lab_topic_name.ipynb`

## Code Style

Lab5 uses `pyproject.toml` (black, isort, pylint, mypy) with line length 120.

## Content Scope (23 Official Lectures + 3 Audio Extensions)

| Chapter | Lectures | Topics |
|---------|----------|--------|
| 0: Intro | L1–L2 | Introduction, NN Basics |
| I: Efficient Inference | L3–L11 | Pruning, Quantization, NAS, KD, MCUNet, TinyEngine |
| II: Domain-Specific | L12–L18 | Transformers/LLMs, ViT, GAN/Video/PointCloud, Diffusion |
| III: Efficient Training | L19–L21 | Distributed, On-Device Training, Fine-tuning |
| IV: Advanced | L22–L23 | Quantum ML |
| Audio Extension (community) | LA1–LA3 | Audio Transformers/ASR, Speech Synthesis, Audio-Language Models |

### Audio Extension (LA1–LA3)

Community-designed notes that apply the course's efficiency principles to audio — a domain the official curriculum doesn't cover:

| Lecture | Topic | Parallel To |
|---------|-------|-------------|
| LA1 | Audio Transformers & ASR (wav2vec, Whisper, Conformer) | L12–L13 |
| LA2 | Speech Synthesis & Audio Generation (FastSpeech, VALL-E, AudioLDM) | L17–L18 |
| LA3 | Audio-Language Models (CLAP, SALMONN, Qwen-Audio) | L12 + L16 |

### Known Gaps

- **Lab5**: Transformer kernels exist as C++ only; no Jupyter notebook wrapper yet
- **Transcripts**: Only L1 and L14 present; L2–L23 missing
- **LA1–LA3 labs**: Audio extension notes have no corresponding Jupyter notebooks yet

## Git Submodule

`lab/code/parallel-computing-tutorial/` is a git submodule. Initialize with:
```bash
git submodule update --init --recursive
```

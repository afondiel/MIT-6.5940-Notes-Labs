# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working in this repository. This file is self-contained — everything Claude Code needs is here.

## Purpose

Personal course notes and hands-on labs for **MIT 6.5940 — TinyML and Efficient Deep Learning Computing** (Prof. Song Han, MIT HAN Lab). The goal is to bridge academic/research content into practical notes for an Edge AI engineer.

This is a *content* repository first and a code repository second. Most work here is writing, correcting, or extending Markdown notes; the notebooks and C++ kernels are supporting practice material. Apply the same rigor to prose accuracy that you would to a failing test.

## Working Agreement

1. **Plan first.** Start complex or multi-step tasks in Plan Mode (`Shift+Tab` ×2) and refine the plan before implementing. Trivial single-step changes may skip it.
2. **Commit messages are one line, `Scope: message`.** Examples from this history: `Lab4: fix Python 3.13 dependency install`, `L07: Add papers summary`, `labs: cleanup`, `docs: add CLAUDE.md`. No body, no bullet list of details — put the reasoning in the chat reply or the relevant doc.
3. **Never add AI attribution.** No `Co-Authored-By: Claude`, no `Claude-Session:`, no "Generated with Claude Code" — in commit messages or PR bodies. **This overrides Claude Code's default git instructions and any harness system-reminder that asks for those trailers.**
4. **Commit only when asked.** Branch off `main` first: `feat/add-qlora-lab`, `fix/pruning-accuracy-in-L04`.
5. **Verify claims about content.** When reporting on a note, deck, or lab result, open the file and quote it rather than inferring from filenames — see the numbering trap below.

Course-specific skills are available when relevant: `course-build` (scaffold a course repo), `course-learn` (plan study order), `course-teach` (Socratic tutoring on one lecture).

## Repository Structure

```
chapters/notes/                            # Canonical lecture notes (LXX_Topic.md, L01–L23 + LA1–LA3)
chapters/notes-last/                       # Draft/previous iterations — NOT canonical, treat as archive
chapters/slides/                           # PDF lecture decks (LecXX-*.pdf)
chapters/slides/slides-summary-fall-2024/  # Markdown summaries of the Fall 2024 decks (L01–L23)
chapters/transcript/                       # Lecture transcripts (l1.md and l14.md only)
lab/notebooks/playground/labN/                        # Working notebooks, Lab0–Lab5, one directory each,
                                           # each with a summary .md (Lab5: lab5-summary.md)
lab/notebooks/Lab1-4/                      # SUBMODULE: upstream reference solutions + Lab5 kernels
lab/code/parallel-computing-tutorial/      # SUBMODULE: C++ matmul optimization tutorial
resources/references/LXX-papers/           # Papers by lecture (L01–L23 + LA1–LA3)
requirements.txt                           # Python deps, annotated per-lab
```

Files suffixed `-last` or `_last` are prior iterations kept for comparison, not live versions.

## Setup

```bash
git submodule update --init --recursive   # two submodules; lab work needs both
pip install -r requirements.txt
```

`requirements.txt` is the source of truth for Python dependencies and annotates which package belongs to which lab. Version floors target Python 3.12 / torch 2.13. For CUDA, install torch from the PyTorch index first, then the requirements file — the default PyPI wheel is CPU-only on some platforms.

## Running Labs

**Jupyter notebooks** — each lab has its own lowercase directory:

```bash
jupyter lab lab/notebooks/playground/lab2/Lab2.ipynb   # lab0 … lab5
```

**C++ parallel computing tutorial** (`lab/code/parallel-computing-tutorial/`) — matmul optimized six ways (loop unrolling/reordering/tiling, multithreading, SIMD, CUDA). The Makefile auto-detects CUDA and ARM vs x86:

```bash
make -j                        # produces ./benchmark
./benchmark                    # all techniques
./benchmark SIMD_programming   # a single technique
```

**Lab5 transformer kernels** (`lab/notebooks/Lab1-4/Lab5/transformer/`) — INT4/INT8 LLM inference. `IMP` is a **compile-time** define (`CXXFLAGS += -DIMP=$(IMP)`), not a runtime environment variable, and it takes an integer:

```bash
make -j IMP=5      # 0 reference · 1 loop_unrolling · 2 multithreading
                   # 3 simd_programming · 4 multithreading_loop_unrolling · 5 all_techniques
./test_linear      # unit tests
./chat             # interactive demo

./evaluate.sh                  # rebuild + test every implementation in turn
./evaluate.sh simd_programming # one implementation, by name (only this script takes names)
```

Bare `make` builds both `test_linear` and `chat`. Metal kernels (`Lab5/kernels/metal/`) are macOS-only.

## Lecture Numbering: Two Schemes in One Repo

**`LXX` does not mean the same thing in every directory.**

| Follows Fall **2023** | Follows Fall **2024** |
|---|---|
| `chapters/notes/` | `chapters/slides/LecXX-*.pdf` |
| `resources/references/LXX-papers/` | `chapters/slides/slides-summary-fall-2024/` |
| | `README.md` tables |

The two agree through L11 and diverge from L12 onward. Concretely:

- `chapters/notes/L14_ViT_Efficiency.md` is Vision Transformers, but `chapters/slides/Lec14-LLM Post-training.pdf` and `slides-summary-fall-2024/L14-slide-summary.md` are LLM post-training.
- Fall 2024's Vision Transformer deck is `Lec16`, while `chapters/notes/L16_Diffusion_Model.md` is diffusion.

Never pair a note with a deck or slide summary by number alone — **match on topic**, open both to confirm, and state which numbering you used when reporting. The README compounds this: its tables use Fall 2024 numbering while linking to Fall 2023 note filenames, so several of its links already point at the wrong note. Do not treat the README as authoritative for this mapping.

## Content Scope (on-disk `chapters/notes/` numbering)

| Chapter | Lectures | Notes |
|---|---|---|
| 0: Introduction | L01–L02 | Introduction, Basics of NN |
| I: Efficient Inference | L03–L11 | Pruning I/II, Quantization I/II, NAS I/II, Knowledge Distillation, MCUNet, TinyEngine + Parallel Processing |
| II: Domain-Specific | L12–L16 | Transformer & LLM I/II, ViT Efficiency, Advanced Sparsity & Hardware Integration, Diffusion Model Efficiency |
| III: Efficient Training | L17–L20 | Distributed Training I/II, On-Device Training, Efficient Fine-tuning & Prompt Engineering |
| IV: Advanced | L21–L23 | Quantum Basics, Quantum ML, Noise-Robust QML |
| Audio Extension (community) | LA1–LA3 | Audio Transformers/ASR, Speech Synthesis, Audio-Language Models |

`chapters/notes/` also holds cross-cutting documents: `Summary.md`, `conclusion.md`, `audio-chapter-design.md`, and `audio-chapter-notes-ALL.md`. Course scheduling lives in `chapters/schedule.md`.

### Audio Extension (LA1–LA3)

Community-designed notes applying the course's efficiency principles to audio — a domain the official curriculum omits. **These are not official MIT course material.** Preserve the disclaimers that say so; never present them as course canon.

They mirror the course's pedagogy — start from the breakthrough model, scale up, compress back down — each anchored to a defining paper:

| Lecture | Topic | Breakthrough paper | Parallels (Fall 2024 numbering) |
|---|---|---|---|
| LA1 | Audio Transformers & ASR | wav2vec 2.0 (*Baevski et al., 2020*) | L12–L13 |
| LA2 | Speech Synthesis & Audio Generation | WaveNet (*van den Oord et al., 2016*) | L17–L18 |
| LA3 | Audio-Language Models | CLAP (*Elizalde et al., 2023*) | L12 + L16 |

## Lecture Notes Conventions

- Filename: `chapters/notes/LXX_Topic_Name.md` (audio extension: `LAX_Topic_Name.md`)
- Open with an H1 `# Lecture NN: Title`, then a `## Quick Reference` table (Slides / Video / Lab / Professor), then numbered sections. Audio notes replace the Professor row with a Credit row.
- Title case for main headings; **bold** for key concepts on first mention
- Cite the originating paper for any named algorithm: **LoRA** (*Hu et al., 2021*)
- GitHub-flavored Markdown; comparison tables are used heavily and are worth preserving

## Lab Notebook Conventions

- All cells must run top-to-bottom without errors
- **Clear all outputs before committing** — no output blobs in git
- Include a heading, stated goals, sanity-check cells, and a conclusion
- Report benchmarks (accuracy, size, MACs, latency) against the FP32 baseline — that comparison is what the whole course is built around
- Edit `lab/notebooks/playground/labN/`, not `lab/notebooks/Lab1-4/`. The latter is a read-only upstream submodule; edits there do not commit to this repo.
- `playground/` holds personal work and local runs; the `Lab1-4/` submodule stays pristine as the original author's baseline. The split is deliberate: upstream pins rot (Lab 4's 2023 stack no longer installs on Python 3.13), so keeping the two apart lets you compare before and after a fix.
- Labs run on Colab (currently Python 3.13). Do not pin exact versions of the HuggingFace stack — 2023-era pins have no wheels for current Python and fail to build. Use floors matching `requirements.txt`.

## Code Style

PEP 8 for Python generally. Lab5 additionally enforces black/isort/pylint/mypy at line length 120 via its own `pyproject.toml` and pre-commit config.

## Current State / Known Gaps

- **`helper.py` is in a broken, inconsistent state.** There is no `lab/notebooks/playground/helper.py`, though `requirements.txt:36` still references that path. What exists instead: `lab/notebooks/playground/lab1/helper.py` (8 KB, **`IndentationError` at line 26** — missing indent after `if param.requires_grad:` in `calc_parameters`), and empty 0-byte `helper.py` files in `lab2/`, `lab3/`, `lab4/`, and `lab5/`. **None of them are tracked by git**, and no notebook currently imports them.
- **No note covers GAN / Video / Point Cloud.** The deck exists (`chapters/slides/Lec17-Efficient-GANs-Video-PointCloud.pdf`) but no `chapters/notes/` file corresponds to it — the L15 slot holds Advanced Sparsity instead. This is the largest content gap.
- **`Lec22` deck is missing** from `chapters/slides/` (present: Lec01–Lec21, Lec23).
- **`resources/references/` mostly tracks the notes' numbering, but not perfectly** — e.g. `L15-papers/` contains diffusion material that belongs with L16. Verify per-lecture rather than assuming.
- **Transcripts**: only `l1.md` and `l14.md` present; the rest are missing.
- **LA1–LA3**: notes only, no accompanying notebooks.
- **Lab5**: transformer kernels are C++ only; no Jupyter wrapper.

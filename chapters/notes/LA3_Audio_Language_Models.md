# Lecture A3: Audio-Language Models and Sound Understanding

## Quick Reference

|Item|Reference|
|---|---|
| Slides | — (community-designed lecture) |
| Video | — |
| Lab | — |
| Credit  | Based on Professor [Song Han](https://github.com/songhan)'s [course framework](https://hanlab.mit.edu/courses/2024-fall-65940) |

## Overview

This lecture covers **audio understanding beyond speech** — general sound event detection, audio tagging, and the emerging field of **Audio-Language Models (ALMs)**. The arc mirrors the multimodal LLM trend in Vision (VLMs like VILA/Flamingo in Lecture 12): first we build **strong audio encoders** through self-supervised pre-training, then we **connect them to LLMs** via contrastive alignment or adapter modules. The breakthrough paper is **CLAP** (*Elizalde et al., 2023*), which does for audio what **CLIP** did for vision — creating a joint audio-text embedding space through contrastive learning. This enables zero-shot audio classification, retrieval, and serves as the bridge connecting audio to large language models.

---

### Slide 1 — Beyond Speech: The World of Sound

Speech is only one type of audio signal. **General audio understanding** covers:

| Domain | Tasks | Applications |
|--------|-------|-------------|
| **Sound Event Detection (SED)** | Identify and timestamp environmental sounds | Security, smart home, wildlife monitoring |
| **Audio Tagging** | Classify audio clips into categories | Content moderation, media indexing |
| **Music Information Retrieval** | Genre, instrument, mood classification | Streaming services, recommendation |
| **Audio Captioning** | Generate natural language descriptions of audio | Accessibility, content search |
| **Audio Question Answering** | Answer questions about audio content | Assistive technology, education |

**Key challenge:** Unlike speech (structured, linguistic), general audio is **highly diverse** — a single model must understand music, environmental sounds, speech, and their mixtures.

---

### Slide 2 — AudioSet: The ImageNet of Audio

**AudioSet** (*Gemmeke et al., 2017*) is the foundational large-scale audio dataset:

- **2.1 million** 10-second YouTube clips.
- **527 audio event classes** in a hierarchical ontology (speech, music, vehicle, animal, etc.).
- Multi-label: each clip can contain **multiple simultaneous events**.

**Impact:** AudioSet did for audio what ImageNet did for vision — it enabled the training of large, general-purpose audio models and became the standard benchmark (measured by mean Average Precision, mAP).

| Model | Year | mAP (AudioSet) | Architecture |
|-------|------|----------------|-------------|
| CNN baseline | 2017 | 0.314 | VGG-like CNN |
| PANNs | 2020 | 0.439 | ResNet-like CNN |
| AST | 2021 | 0.485 | ViT (Transformer) |
| BEATs | 2022 | 0.538 | Iterative audio pre-training |
| CLAP | 2023 | — (zero-shot) | Contrastive audio-text |

> **Reference:** Gemmeke et al., *"Audio Set: An Ontology and Human-Labeled Dataset for Audio Events"* (ICASSP 2017)

---

### Slide 3 — PANNs: Large-Scale Pre-Trained Audio Neural Networks

**PANNs** (*Kong et al., 2020*) established the paradigm of **pre-trained audio feature extractors**:

- Various CNN architectures (CNN6, CNN10, CNN14, ResNet38) trained on AudioSet.
- Released as **pre-trained feature extractors** — analogous to ImageNet-pre-trained ResNets for vision.
- CNN14 (80M params) became the standard audio backbone for years.

**Limitation:** Purely supervised — requires labeled AudioSet data. Does not capture the rich unlabeled audio available online.

> **Reference:** Kong et al., *"PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition"* (IEEE/ACM TASLP, 2020)

---

### Slide 4 — BEATs: Audio Pre-Training with Acoustic Tokenizers

**BEATs** (*Chen et al., 2022*) introduced **iterative self-supervised** pre-training for audio:

**Two-stage iterative process:**
1. **Acoustic Tokenizer:** A separate model learns to discretize Mel-spectrograms into **semantic tokens** (like HuBERT for speech, but for general audio).
2. **Audio Encoder (ViT):** Trained via masked prediction to predict the acoustic tokens of masked patches.
3. **Iterate:** The encoder's representations are used to train a better tokenizer → train a better encoder → repeat.

**Result:** State-of-the-art **0.538 mAP** on AudioSet — surpassing AST and all previous approaches. The iterative bootstrap between tokenizer and encoder is the key insight.

**Parallel to HuBERT (Lecture A1):** BEATs generalizes HuBERT's idea from speech-only to all audio categories. The same principle: discover discrete targets → predict masked positions.

> **Reference:** Chen et al., *"BEATs: Audio Pre-Training with Acoustic Tokenizers"* (ICML 2023)

---

### Slide 5 — CLAP: Contrastive Language-Audio Pre-training (THE Breakthrough)

**CLAP** (*Elizalde et al., 2023*) is the **CLIP moment for audio**:

**Architecture:**
- **Audio Encoder:** CNN14 or HTSAT (Hierarchical Token-Semantic Audio Transformer) — encodes audio into an embedding vector.
- **Text Encoder:** RoBERTa or BERT — encodes text descriptions into an embedding vector.
- **Contrastive Training:** Audio-text pairs are trained so that matching pairs have high cosine similarity and non-matching pairs have low similarity (InfoNCE loss, same as CLIP).

**Training data:** 633K audio-text pairs from AudioSet (with captions), Clotho, AudioCaps, and web-scraped data.

**Capabilities enabled:**
1. **Zero-shot audio classification:** Classify audio into ANY categories by comparing audio embedding with text embeddings of class names — no task-specific training needed.
2. **Audio-text retrieval:** Find audio clips matching a text query and vice versa.
3. **Conditioning for generation:** AudioLDM (Lecture A2) uses CLAP embeddings for text-to-audio generation.

**Why this matters:** CLAP creates the **bridge** between the audio domain and language, enabling all downstream audio-language applications.

> **References:**
> - Elizalde et al., *"CLAP: Learning Audio Concepts from Natural Language Supervision"* (ICASSP 2023)
> - Wu et al., *"Large-Scale Contrastive Language-Audio Pretraining with Feature Fusion and Keyword-to-Caption Augmentation"* (ICASSP 2023)

---

### Slide 6 — Architecture Patterns for Audio-Language Models

ALMs connect a **frozen audio encoder** to a **frozen or fine-tuned LLM**. Three main patterns (mirroring VLM designs from Lecture 12):

| Pattern | Description | Example |
|---------|-------------|---------|
| **Projection** | Linear layer maps audio embeddings → LLM input space | LTU, Qwen-Audio |
| **Q-Former / Adapter** | Learnable query tokens cross-attend to audio features | Pengi, SALMONN |
| **Interleaved tokens** | Audio tokens inserted into the text token sequence | AudioPaLM |

**Common recipe:**
1. **Audio Encoder** (frozen): wav2vec 2.0, BEATs, Whisper encoder, or CLAP audio encoder.
2. **Bridge module** (trained): Projection layer, Q-Former, or linear adapter.
3. **LLM backbone** (frozen or LoRA-tuned): LLaMA, Vicuna, OPT.

**Key efficiency insight:** By freezing both the audio encoder and the LLM, only the **bridge module** needs training — typically <5% of total parameters. This is the audio equivalent of **LoRA/Adapter** (Lecture 13).

---

### Slide 7 — Pengi: Audio Language Model for Audio Tasks

**Pengi** (*Deshmukh et al., 2023*):

- Audio encoder: **CLAP audio encoder** (frozen).
- Bridge: audio features are **projected and concatenated** with text prefix tokens.
- LLM: **GPT-2** (fine-tuned).
- **All audio tasks are reformulated as text generation:**
  - Audio captioning: "Describe this audio" → "A dog barking in a park"
  - Audio QA: "What instrument is playing?" → "Piano"
  - Sound event detection: "What sounds are present?" → "Traffic, birds, speech"

**Result:** First model to unify diverse audio tasks under a single generative framework, achieving competitive results across 21 audio benchmarks.

> **Reference:** Deshmukh et al., *"Pengi: An Audio Language Model for Audio Tasks"* (NeurIPS 2023)

---

### Slide 8 — LTU: Listen, Think, and Understand

**LTU** (*Gong et al., 2023*):

- Audio encoder: **AST** (Audio Spectrogram Transformer, frozen).
- Bridge: **Linear projection** + learnable audio tokens.
- LLM: **LLaMA** (LoRA fine-tuned).
- Trained on a **large-scale audio QA dataset** generated by prompting GPT-4 with AudioSet captions.
- Supports **open-ended audio reasoning**: "Why do you think this audio was recorded outdoors?" → generates multi-sentence explanation.

**Key contribution:** Demonstrated that **LLM reasoning capabilities** (from text pre-training) can be applied to audio understanding through simple projection — the audio encoder provides perception, the LLM provides reasoning.

> **Reference:** Gong et al., *"Listen, Think, and Understand"* (ICLR 2024)

---

### Slide 9 — SALMONN: Generic Hearing for LLMs

**SALMONN** (*Tang et al., 2023*): Speech Audio Language Music Open Neural Network.

**Key innovation: Dual audio encoder.**
- **Whisper encoder** (frozen): Captures **speech-specific** features (text content, speaker characteristics).
- **BEATs encoder** (frozen): Captures **non-speech audio** features (music, environmental sounds).
- **Window-level Q-Former** (trained): Cross-attends to both encoder outputs and produces a compact set of audio tokens.
- **LLM:** Vicuna (LoRA fine-tuned).

**Why dual encoder?** Speech and non-speech audio have fundamentally different acoustic properties. A single encoder optimized for one domain under-performs on the other. The dual-encoder design covers the full audio spectrum.

**Capabilities:** Speech recognition, audio captioning, music analysis, sound event detection, multilingual ASR, audio QA — **all in one model**.

> **Reference:** Tang et al., *"SALMONN: Towards Generic Hearing Abilities for Large Language Models"* (ICLR 2024)

---

### Slide 10 — Qwen-Audio: Scaling Audio Understanding

**Qwen-Audio** (*Chu et al., 2023*):

- Audio encoder: **Whisper-large encoder** (fine-tuned).
- Bridge: **Linear projection**.
- LLM: **Qwen-7B** (fine-tuned).
- Trained on **30+ audio tasks** across speech, music, and environmental sound.
- No task-specific heads — **unified as text generation**.

**Qwen2-Audio** (2024) extended this with:
- Voice interaction mode: process interleaved audio and text instructions.
- Direct audio chat without text transcription intermediary.

**Scaling insight:** Like Whisper scaling for ASR (Lecture A1), Qwen-Audio shows that **more data + larger LLM backbone** consistently improves audio understanding — but at massive compute cost, motivating efficient deployment.

> **Reference:** Chu et al., *"Qwen-Audio: Advancing Universal Audio Understanding via Unified Large-Scale Audio-Language Models"* (2023)

---

### Slide 11 — AudioPaLM: Unified Speech and Text

**AudioPaLM** (*Rubenstein et al., 2023*):

- Merges **PaLM-2** (text LLM) with **AudioLM** (audio generation) into a single model.
- Input/output: both **text tokens** and **audio tokens** (from SoundStream/EnCodec).
- A single model handles: ASR, TTS, speech-to-speech translation, voice cloning.

**Key insight:** Since neural audio codecs (Lecture A2) convert audio to discrete tokens, audio can be treated as **just another language** in a multilingual LLM. The efficiency techniques for LLMs (Lectures 12–15) — KV cache, GQA, quantization, speculative decoding — all apply directly.

> **Reference:** Rubenstein et al., *"AudioPaLM: A Large Language Model That Can Speak and Listen"* (2023)

---

### Slide 12 — Efficiency for Audio-Language Models

ALMs inherit the **same efficiency challenges as VLMs and LLMs**:

**Audio Encoder Efficiency:**
- The audio encoder typically runs on 10–30 second clips → long sequences (~3,000 tokens for Whisper encoder at 30s).
- **Subsampling:** Aggressive stride in the audio encoder reduces token count.
- **Token pruning:** Prune low-information audio tokens (silence, background noise) before feeding to the LLM — analogous to token pruning in ViTs (Lecture 16).

**LLM Backbone Efficiency:**
- All techniques from Lectures 12–15 apply:
  - **Quantization (AWQ/GPTQ):** Quantize the LLM backbone to INT4.
  - **LoRA:** Fine-tune only the bridge module + LoRA adapters.
  - **KV Cache optimization (GQA, PageAttention):** Critical for long audio inputs.
  - **Speculative decoding:** Use a small draft model for text generation.

**Inference Pipeline:**
```
Audio (10s) → [Encoder: ~50ms] → [Bridge: ~5ms] → [LLM generation: ~500ms]
                    5%                 <1%               95%
```
The LLM dominates — so **LLM efficiency is the priority**.

---

### Slide 13 — TinyML for Audio: Sound Classification on MCU

Beyond LLM-scale models, audio understanding also runs at the **extreme edge**:

**Sound Event Detection on Microcontrollers:**
- Target: Cortex-M4/M7, <512KB RAM.
- Architecture: Small CNN (DS-CNN, MobileNet-v1) on Mel-spectrograms.
- Use case: Gunshot detection, glass breaking, baby crying, cough detection.

**Micro-NAS for Audio:**
- Apply MCUNet-style NAS (Lecture 10) with audio-specific search space.
- **Key difference from image NAS:** Audio spectrograms have very different aspect ratios (wide, short) — the search space must accommodate this.

**On-Device Keyword Spotting + Wake Word:**
- The most deployed TinyML audio application — billions of devices.
- 2-stage: always-on KWS (MCU, <1mW) → triggered full ASR (application processor).
- **Quantization:** INT8 and even binary/ternary weights for the KWS model to fit in 64KB Flash.

> **Reference:** Banbury et al., *"MLPerf Tiny Benchmark"* (NeurIPS 2021)

---

### Slide 14 — Edge Deployment: On-Device ASR, TTS, and Audio Understanding

**Full Audio Pipeline on Device:**

| Component | Model | Size (INT8) | Latency (ARM) |
|-----------|-------|-------------|----------------|
| Wake Word | DS-CNN (NAS) | 64 KB | <10ms |
| ASR | Distil-Whisper-small | ~150 MB | ~300ms / utterance |
| TTS | VITS-small (pruned) | ~30 MB | ~50ms / sentence |
| Audio Classification | MobileNet-v2 | ~5 MB | ~20ms / clip |

**Key insight:** The audio modality is **uniquely suited for edge deployment** because:
1. Audio data rates are low (~16KB/s at 16kHz, 16-bit) vs. video (~100MB/s).
2. Privacy concerns are critical (voice data).
3. Latency requirements are tight (real-time conversation: <200ms round-trip).

**Combined quantization + pruning + distillation** (Lectures 3–9) can bring the full pipeline under 200MB — feasible for modern smartphones.

---

### Slide 15 — Summary: The Audio-Language Arc

| Era | Model | Key Innovation | Paradigm |
|-----|-------|---------------|----------|
| Supervised | PANNs (2020) | Pre-trained CNN on AudioSet | Classification |
| Transformer | AST (2021) | ViT for audio spectrograms | Discriminative |
| Self-Supervised | BEATs (2022) | Iterative tokenizer + masked prediction | Foundation model |
| **Contrastive** | **CLAP** (2023) | Audio-text alignment (like CLIP) | Zero-shot |
| **ALM** | Pengi, LTU, SALMONN (2023) | Audio encoder → LLM bridge | Generative reasoning |
| Unified | AudioPaLM, Qwen-Audio (2023) | Single model for speech + audio + text | Multimodal LLM |
| **Edge** | TinyML, Distil-Whisper | NAS + KD + Quantization | On-device |

**The pattern mirrors Vision-Language Models exactly:**
- CLIP (vision-text) → CLAP (audio-text)
- LLaVA/VILA (VLM) → LTU/SALMONN (ALM)
- ViT efficiency → AST/BEATs efficiency
- MCUNet (vision TinyML) → KWS/SED TinyML

---

## References

- Gemmeke et al., *"Audio Set: An Ontology and Human-Labeled Dataset for Audio Events"* (ICASSP 2017)
- Kong et al., *"PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition"* (2020)
- Chen et al., *"BEATs: Audio Pre-Training with Acoustic Tokenizers"* (ICML 2023)
- Elizalde et al., *"CLAP: Learning Audio Concepts from Natural Language Supervision"* (ICASSP 2023)
- Deshmukh et al., *"Pengi: An Audio Language Model for Audio Tasks"* (NeurIPS 2023)
- Gong et al., *"Listen, Think, and Understand"* (ICLR 2024)
- Tang et al., *"SALMONN: Towards Generic Hearing Abilities for Large Language Models"* (ICLR 2024)
- Chu et al., *"Qwen-Audio: Advancing Universal Audio Understanding"* (2023)
- Rubenstein et al., *"AudioPaLM: A Large Language Model That Can Speak and Listen"* (2023)
- Wu et al., *"Large-Scale Contrastive Language-Audio Pretraining"* (ICASSP 2023)
- Banbury et al., *"MLPerf Tiny Benchmark"* (NeurIPS 2021)

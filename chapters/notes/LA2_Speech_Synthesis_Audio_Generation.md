# Lecture A2: Efficient Speech Synthesis and Audio Generation

## Quick Reference

|Item|Reference|
|---|---|
| Slides | — (community-designed lecture) |
| Video | — |
| Lab | — |
| Credit  | Based on Professor [Song Han](https://github.com/songhan)'s [course framework](https://hanlab.mit.edu/courses/2024-fall-65940) |

## Overview

This lecture covers the **generative side** of audio: how do we synthesize speech and generate audio efficiently? The arc mirrors Vision's generative story (Lectures 17–18): a pioneering **autoregressive model creates a breakthrough in quality** but is **impossibly slow** — then the field races to find parallel, efficient alternatives. The breakthrough paper is **WaveNet** (*van den Oord et al., 2016*), which for the first time generated **human-quality speech** using a neural network, but required generating audio **one sample at a time** (24,000 sequential steps per second of audio). Everything that follows — FastSpeech, VITS, HiFi-GAN, neural codecs, audio diffusion — is the field's response to WaveNet's latency problem.

---

### Slide 1 — The TTS Pipeline: Text to Waveform

Modern Text-to-Speech (TTS) typically follows a two-stage pipeline:

```
Text → [Text Encoder] → [Acoustic Model] → Mel-Spectrogram → [Vocoder] → Waveform
```

| Stage | Role | Compute Profile |
|-------|------|----------------|
| **Acoustic Model** | Predicts Mel-spectrogram from text/phonemes | Moderate (sequence-to-sequence) |
| **Vocoder** | Converts Mel-spectrogram to raw waveform | **Heavy** (generates 16K–48K samples/sec) |

The vocoder is the **bottleneck** — it must produce tens of thousands of audio samples per second of speech. This is fundamentally harder than text generation (a few tokens per second) or image generation (a single output).

---

### Slide 2 — Pre-Neural TTS: Concatenative and Parametric

**Before deep learning:**
- **Concatenative TTS:** Stitches together pre-recorded speech segments from a large database. High quality for covered domains but **inflexible** and requires massive storage.
- **Parametric TTS (HMM-based):** Uses statistical models to generate vocoder parameters. Lightweight but sounds **robotic** — the so-called "machine voice."

**The gap:** Neither approach could produce natural, expressive speech. This motivated neural approaches.

---

### Slide 3 — WaveNet: The Autoregressive Breakthrough

**WaveNet** (*van den Oord et al., 2016*) from DeepMind was the first neural model to generate **near-human-quality** speech:

**Architecture:**
- **Dilated causal convolutions** stacked in layers with exponentially increasing dilation rates (1, 2, 4, 8, ..., 512).
- Each layer's receptive field doubles, covering thousands of past samples.
- **Autoregressive:** Predicts each audio sample conditioned on ALL previous samples: $p(x_t | x_1, ..., x_{t-1})$.
- Uses **μ-law companding** to quantize 16-bit audio into 256 bins, modeled as a 256-way softmax.

**Impact:** MOS (Mean Opinion Score) jumped from ~3.5 (parametric TTS) to **~4.2** (close to human ~4.5).

**The fatal flaw:** Generating 1 second of audio at 16kHz requires **16,000 sequential forward passes**. Real-time factor: **~1000× slower than real-time** on GPU. Completely impractical for deployment.

> **Reference:** van den Oord et al., *"WaveNet: A Generative Model for Raw Audio"* (2016)

---

### Slide 4 — Parallel WaveNet: Distilling Speed from Quality

**Parallel WaveNet** (*van den Oord et al., 2017*) applied **Knowledge Distillation** (Lecture 9) to solve WaveNet's speed problem:

- **Teacher:** Pre-trained autoregressive WaveNet (slow, high quality).
- **Student:** An **Inverse Autoregressive Flow (IAF)** — generates all samples in **parallel** in a single forward pass.
- **Training:** The student is trained to match the teacher's output distribution using probability density distillation.

**Result:** **1000× speedup** — real-time speech synthesis on a single GPU.

**Key insight (Song Han perspective):** This is one of the most dramatic demonstrations of Knowledge Distillation in practice — converting an autoregressive model into a parallel one, trading training complexity for inference speed. The same pattern appears in diffusion model distillation (Lecture 18).

> **Reference:** van den Oord et al., *"Parallel WaveNet: Fast High-Fidelity Speech Synthesis"* (ICML 2018)

---

### Slide 5 — Tacotron 2: End-to-End TTS

**Tacotron 2** (*Shen et al., 2018*) introduced **end-to-end** neural TTS:

```
Characters → [Encoder (Conv + BiLSTM)] → [Attention] → [Decoder (autoregressive)] → Mel-Spectrogram → [WaveNet Vocoder] → Waveform
```

- The acoustic model is a **sequence-to-sequence** architecture with **location-sensitive attention**.
- Decoder is **autoregressive** — predicts one Mel-spectrogram frame at a time.
- Achieved **MOS 4.53** — virtually indistinguishable from human speech.

**Efficiency concern:** Two-stage (Tacotron 2 + WaveNet) and both stages are autoregressive → compound latency problem.

> **Reference:** Shen et al., *"Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions"* (ICASSP 2018)

---

### Slide 6 — FastSpeech: The Non-Autoregressive Revolution

**FastSpeech** (*Ren et al., 2019*) is the **efficiency breakthrough** for TTS — the equivalent of removing the autoregressive bottleneck:

**Key innovation: Duration Predictor**
- A small network predicts **how many Mel frames** each phoneme should span.
- The input phoneme sequence is **expanded** to match the output length.
- The Transformer decoder then generates all Mel frames **in parallel**.

**Architecture:**
```
Phonemes → [Feed-Forward Transformer Encoder] → [Duration Predictor + Length Regulator] → [Feed-Forward Transformer Decoder] → Mel-Spectrogram (parallel)
```

**Results:**
- **270× faster** Mel-spectrogram generation than autoregressive Tacotron 2.
- Eliminates word-skipping and repeating errors (robustness).
- Controllable speed (stretch/compress durations).

> **Reference:** Ren et al., *"FastSpeech: Fast, Robust and Controllable Text to Speech"* (NeurIPS 2019)

---

### Slide 7 — FastSpeech 2: Removing the Teacher Dependency

**FastSpeech 2** (*Ren et al., 2021*) improved upon FastSpeech:

- Trains **directly on ground-truth** durations (from forced alignment) rather than distilling from an autoregressive teacher.
- Adds **variance adaptors** for pitch, energy, and duration — enabling expressive and controllable synthesis.
- Uses **direct waveform loss** in FastSpeech 2s variant — bypassing the Mel-spectrogram stage entirely.

**Result:** Simpler training pipeline, better quality, and **3× faster training** than FastSpeech while maintaining the same fast parallel inference.

> **Reference:** Ren et al., *"FastSpeech 2: Fast and High-Quality End-to-End Text to Speech"* (ICLR 2021)

---

### Slide 8 — GAN-Based Vocoders: HiFi-GAN

The vocoder (Mel → waveform) was still a bottleneck. **HiFi-GAN** (*Kong et al., 2020*) solved it using Generative Adversarial Networks:

**Generator:**
- Transposed convolutions for upsampling Mel-spectrogram to waveform sample rate.
- **Multi-Receptive Field Fusion (MRF):** Parallel residual blocks with different kernel sizes to capture patterns at multiple time scales.

**Discriminator:**
- **Multi-Period Discriminator (MPD):** Evaluates different periodic patterns in audio.
- **Multi-Scale Discriminator (MSD):** Evaluates audio at different resolutions.

**Result:**
- Synthesis speed: **167× faster than real-time** on GPU, **13× faster on CPU**.
- Quality matches autoregressive WaveNet.
- Model size: ~14M parameters — small enough for mobile deployment.

**Key parallel to GAN Compression (Lecture 17):** Like GauGAN compression, the same teacher-student and architectural efficiency principles apply to audio GANs.

> **Reference:** Kong et al., *"HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis"* (NeurIPS 2020)

---

### Slide 9 — VITS: Fully End-to-End TTS

**VITS** (*Kim et al., 2021*) unified the acoustic model and vocoder into a **single end-to-end** model:

**Architecture:**
- **Text Encoder:** Transformer.
- **Posterior Encoder:** Extracts latent representations from ground-truth audio during training.
- **Flow-based Decoder:** Normalizing flows transform simple distributions into complex speech distributions.
- **HiFi-GAN Decoder:** Generates waveform from latent representations.
- **Stochastic Duration Predictor:** Models duration distributions, enabling diverse prosody.
- **Training:** Variational inference + adversarial training (VAE + GAN hybrid).

**Result:** End-to-end: text in → waveform out. No separate vocoder. Naturalness surpasses two-stage systems. Enables **single-model deployment** — critical for edge devices.

> **Reference:** Kim et al., *"Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech"* (ICML 2021)

---

### Slide 10 — Neural Audio Codecs: SoundStream and EnCodec

A paradigm shift: treat audio as **discrete tokens** (like text), enabling LLM-style modeling.

**SoundStream** (*Zeghidour et al., 2021*):
- **Encoder:** Convolutional, compresses 24kHz audio to 50Hz latent frames.
- **Residual Vector Quantization (RVQ):** Each frame is quantized using a **cascade of codebooks** — each codebook refines the residual error from the previous.
- **Decoder:** Convolutional, reconstructs waveform from quantized codes.
- Achieves **high-quality audio at 3–18 kbps** (vs. 128 kbps for MP3).

**EnCodec** (*Défossez et al., 2022*):
- Builds on SoundStream with **multi-scale STFT discriminator** and entropy-based bitrate control.
- Supports **1.5–24 kbps** at 24–48 kHz.

**Why this matters:** RVQ tokens convert continuous audio into a **discrete sequence** — enabling the entire arsenal of LLM efficiency techniques (KV cache, quantization, speculative decoding) to be applied to audio generation.

> **References:**
> - Zeghidour et al., *"SoundStream: An End-to-End Neural Audio Codec"* (IEEE/ACM TASLP, 2021)
> - Défossez et al., *"High Fidelity Neural Audio Compression"* (TMLR, 2022)

---

### Slide 11 — VALL-E: Voice Cloning as Language Modeling

**VALL-E** (*Wang et al., 2023*) reframes TTS as a **neural codec language model**:

- Input: text tokens + **3-second audio prompt** (encoded by EnCodec into codec tokens).
- The model is a **decoder-only Transformer** (like GPT) that generates **codec tokens** autoregressively.
- The EnCodec decoder then converts tokens back to waveform.

**Result:** **Zero-shot voice cloning** — 3 seconds of any speaker's voice is enough to synthesize speech in that voice, without any fine-tuning.

**Efficiency challenge:** VALL-E generates audio tokens autoregressively → inherits the same latency problems as LLM text generation. All LLM efficiency techniques from Lectures 12–15 (KV cache, speculative decoding, quantization) directly apply.

> **Reference:** Wang et al., *"VALL-E: Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers"* (2023)

---

### Slide 12 — SoundStorm: Efficient Parallel Audio Token Generation

**SoundStorm** (*Borsos et al., 2023*) addresses VALL-E's autoregressive bottleneck:

- Uses **MaskGIT-style parallel decoding** — iteratively unmasks audio tokens over a small number of rounds (8–16), rather than generating each token sequentially.
- Generates **RVQ tokens level by level** — coarse tokens first (low codebook), then progressively refines with finer codebooks.
- **100× faster** than autoregressive audio token generation.
- Produces **30 seconds of audio in 0.5 seconds** on a TPU.

**Key parallel to Diffusion efficiency (Lecture 18):** Just as DDIM/DPM-Solver reduce diffusion steps, SoundStorm reduces autoregressive audio generation from thousands of sequential steps to ~16 parallel rounds.

> **Reference:** Borsos et al., *"SoundStorm: Efficient Parallel Audio Generation"* (ICML 2023)

---

### Slide 13 — Audio Diffusion Models: DiffWave and AudioLDM

**Audio Diffusion** applies the diffusion framework (Lecture 18) to audio generation.

**DiffWave** (*Kong et al., 2020*):
- Diffusion model operating on **raw waveforms**.
- Uses **dilated convolutions** (like WaveNet) as the denoising backbone — but the iterative refinement is over **diffusion steps**, not autoregressive time steps.
- Non-autoregressive: all waveform samples are generated simultaneously at each step.
- Quality matches autoregressive models; **speed depends on number of diffusion steps**.

**AudioLDM** (*Liu et al., 2023*):
- Applies **Latent Diffusion** (like Stable Diffusion for images) to audio.
- Operates in a compressed **latent space** from a VAE trained on Mel-spectrograms.
- Uses **CLAP text embeddings** (see Lecture A3) for text-to-audio conditioning.
- Supports: text-to-audio, text-to-music, audio inpainting, style transfer.

**Efficiency:** Same techniques from Lecture 18 apply — sampling acceleration (fewer steps), classifier-free guidance, latent space compression.

> **References:**
> - Kong et al., *"DiffWave: A Versatile Diffusion Model for Audio Synthesis"* (ICLR 2021)
> - Liu et al., *"AudioLDM: Text-to-Audio Generation with Latent Diffusion Models"* (ICML 2023)

---

### Slide 14 — NAS for TTS: LightSpeech

**LightSpeech** (*Luo et al., 2021*) applies **Neural Architecture Search** (Lectures 7–8) to find efficient TTS architectures:

- Search space: Transformer block configurations (number of heads, FFN dimensions, number of layers) for both encoder and decoder.
- **Hardware-aware NAS** with latency constraints — exactly the same methodology as ProxylessNAS/OFA from the course, adapted for audio.
- Discovered architectures are **asymmetric** — deeper encoder, shallower decoder — matching the intuition that understanding text is harder than generating Mel-spectrograms.

**Result:** **6.5× compression** over FastSpeech with no significant quality degradation on LJSpeech.

> **Reference:** Luo et al., *"LightSpeech: Lightweight and Fast Text to Speech with Neural Architecture Search"* (ICASSP 2021)

---

### Slide 15 — Pruning and Quantization for TTS

**Pruning:**
- Structured pruning of Transformer layers in FastSpeech 2 / VITS.
- **Finding:** TTS models are highly over-parameterized — up to **50% of attention heads can be removed** with <0.05 MOS degradation.
- HiFi-GAN vocoder: channel pruning reduces model from 14M → 3.5M parameters while maintaining quality.

**Quantization:**
- INT8 quantization of VITS achieves **1.8× speedup** on mobile ARM CPUs.
- **Mixed-precision:** Encoder layers tolerate INT8 well; decoder layers (especially the flow-based components) are more sensitive — FP16 recommended.
- HiFi-GAN INT8: quality maintained with **2× memory reduction**.

**Knowledge Distillation:**
- Teacher: full VITS → Student: pruned + quantized VITS-small.
- The student is trained to match intermediate representations (not just output) — **feature-level KD** (Lecture 9).

> **Reference:** Huang et al., *"Towards Lightweight Transformer-based TTS"* (InterSpeech 2022)

---

### Slide 16 — Summary: The Audio Generation Arc

| Era | Model | Innovation | Latency (1s audio) |
|-----|-------|-----------|-------------------|
| Pre-Neural | Concatenative, HMM | Database / statistical | Real-time |
| **Autoregressive** | **WaveNet** (2016) | Dilated causal conv, sample-by-sample | ~1000s (GPU) |
| Distilled | Parallel WaveNet (2017) | IAF + KD from WaveNet | ~1ms |
| End-to-End | Tacotron 2 (2018) | Seq2seq TTS + WaveNet vocoder | ~10s |
| **Non-AR** | **FastSpeech** (2019) | Duration predictor, parallel Mel gen | ~10ms |
| GAN Vocoder | HiFi-GAN (2020) | Multi-scale/period discriminators | <1ms (GPU) |
| Unified | VITS (2021) | VAE + flow + GAN, single model | ~5ms |
| **Codec LM** | **VALL-E** (2023) | Zero-shot voice cloning via codec tokens | ~seconds (AR) |
| Parallel Codec | SoundStorm (2023) | MaskGIT parallel decoding | ~0.5s / 30s audio |
| **Diffusion** | AudioLDM (2023) | Latent diffusion for audio | ~seconds (tunable) |

**The pattern mirrors Vision Generation exactly:** Autoregressive breakthrough → parallel alternatives → GANs → Diffusion → LM-based approaches.

---

## References

- van den Oord et al., *"WaveNet: A Generative Model for Raw Audio"* (2016)
- van den Oord et al., *"Parallel WaveNet: Fast High-Fidelity Speech Synthesis"* (ICML 2018)
- Shen et al., *"Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions"* (ICASSP 2018)
- Ren et al., *"FastSpeech: Fast, Robust and Controllable Text to Speech"* (NeurIPS 2019)
- Ren et al., *"FastSpeech 2: Fast and High-Quality End-to-End Text to Speech"* (ICLR 2021)
- Kong et al., *"HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis"* (NeurIPS 2020)
- Kim et al., *"Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech"* (ICML 2021)
- Zeghidour et al., *"SoundStream: An End-to-End Neural Audio Codec"* (2021)
- Défossez et al., *"High Fidelity Neural Audio Compression"* (2022)
- Wang et al., *"VALL-E: Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers"* (2023)
- Borsos et al., *"SoundStorm: Efficient Parallel Audio Generation"* (ICML 2023)
- Kong et al., *"DiffWave: A Versatile Diffusion Model for Audio Synthesis"* (ICLR 2021)
- Liu et al., *"AudioLDM: Text-to-Audio Generation with Latent Diffusion Models"* (ICML 2023)
- Luo et al., *"LightSpeech: Lightweight and Fast Text to Speech with Neural Architecture Search"* (ICASSP 2021)


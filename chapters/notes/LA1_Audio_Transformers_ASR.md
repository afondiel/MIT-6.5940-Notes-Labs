# Lecture A1: Audio Transformers and Efficient Speech Recognition

## Quick Reference

|Item|Reference|
|---|---|
| Slides | — (community-designed lecture) |
| Video | — |
| Lab | — |
| Credit  | Based on Professor [Song Han](https://github.com/songhan)'s [course framework](https://hanlab.mit.edu/courses/2024-fall-65940) |

## Overview

This lecture moves from general efficient inference techniques to **audio-specific** optimization, focusing on the **Transformer architecture's impact on speech and audio understanding**. We trace the arc from traditional signal-processing pipelines through RNN-based end-to-end models to the self-supervised Transformer revolution—mirroring the RNN→Transformer shift in NLP (Lecture 12). The breakthrough paper is **wav2vec 2.0** (*Baevski et al., 2020*), which did for speech what BERT did for text: large-scale self-supervised pre-training that eliminates the need for massive labeled datasets.

---

### Slide 1 — Why Audio Matters for Efficient AI

Audio is **ubiquitous** on edge devices: smartphones, smart speakers, earbuds, hearing aids, cars, and IoT sensors. Key tasks include:

- **Automatic Speech Recognition (ASR):** Voice assistants, dictation, transcription.
- **Keyword Spotting (KWS):** "Hey Siri", "OK Google" — runs on always-on microcontrollers.
- **Sound Event Detection (SED):** Security, health monitoring, wildlife, industrial fault detection.
- **Speaker Verification / Identification:** Biometric authentication.

The core challenge: audio models must often run **on-device** for privacy and latency reasons, under tight compute and memory budgets.

---

### Slide 2 — Audio Representations: From Waveform to Spectrogram

Before any neural network, raw audio must be represented:

| Representation | Description | Dimensions (1s @ 16kHz) |
|---|---|---|
| **Raw Waveform** | 1D signal, amplitude vs. time | 16,000 samples |
| **Spectrogram** | STFT magnitude, 2D time-frequency | ~100 × 257 |
| **Mel-Spectrogram** | Spectrogram on perceptual Mel scale | ~100 × 80 |
| **MFCC** | Cepstral coefficients from Mel-spectrogram | ~100 × 13 |

- Mel-spectrograms are the **dominant input** for modern neural audio models—they compress frequency resolution to match human perception while retaining enough detail.
- **Key insight:** A Mel-spectrogram is essentially a **single-channel image**, which is why Vision Transformer ideas transfer directly to audio.

> **Reference:** Davis & Mermelstein, *"Comparison of Parametric Representations for Monosyllabic Word Recognition in Continuously Spoken Sentences"* (1980)

---

### Slide 3 — Traditional ASR: The HMM–DNN Pipeline

Before deep learning, ASR used a **modular pipeline**:

1. **Feature Extraction** → MFCCs
2. **Acoustic Model** → HMM + GMM (later replaced by DNNs)
3. **Pronunciation Model** → Phoneme-to-word lexicon
4. **Language Model** → N-gram or WFST

**Limitations:**
- Each module trained **independently** — error propagation between stages.
- Requires **expert-designed phoneme inventories** and large pronunciation dictionaries.
- Hard to scale to new languages without linguistic expertise.

The **DNN-HMM hybrid** (*Hinton et al., 2012*) replaced the GMM with a deep neural network for acoustic modeling, dramatically improving accuracy while keeping the HMM decoding framework.

> **Reference:** Hinton et al., *"Deep Neural Networks for Acoustic Modeling in Speech Recognition"* (IEEE Signal Processing Magazine, 2012)

---

### Slide 4 — End-to-End ASR: CTC and Sequence-to-Sequence

The breakthrough toward simplicity: **end-to-end** models that map audio directly to text.

**Connectionist Temporal Classification (CTC):**
- Introduces a **blank token** and marginalizes over all valid alignments between input frames and output characters.
- Enables training without frame-level alignment annotations.
- Output assumption: tokens are **conditionally independent** given the input — limits modeling power.

**Sequence-to-Sequence with Attention (Listen, Attend, Spell / LAS):**
- Encoder processes audio, decoder autoregressively generates text with attention over encoder states.
- Models output dependencies — better accuracy than CTC alone.
- **Problem:** Autoregressive decoding is **sequential** and slow.

> **References:**
> - Graves et al., *"Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks"* (ICML 2006)
> - Chan et al., *"Listen, Attend and Spell"* (ICASSP 2016)

---

### Slide 5 — Scaling RNN-based ASR: Deep Speech 2

**Deep Speech 2** (*Amodei et al., 2015*) demonstrated that ASR could be treated as a **scaling problem**: stack more layers, use more data, train on GPUs.

- Architecture: Multiple layers of **bidirectional RNNs** (GRUs) with batch normalization.
- Trained on **12,000 hours** of speech — massive for 2015.
- Achieved near-human performance on certain benchmarks.

**But the fundamental limitation remained:** RNNs are **sequential**. Like in NLP (Lecture 12), the recurrence creates:
- Non-parallelizable training and inference.
- Vanishing gradients over long utterances.
- Memory bottleneck for long audio (a 30s utterance at 100 frames/s = 3,000 time steps).

> **Reference:** Amodei et al., *"Deep Speech 2: End-to-End Speech Recognition in English and Mandarin"* (ICML 2016)

---

### Slide 6 — The Transformer Arrives in Speech

**Speech-Transformer** directly applied the encoder-decoder Transformer to ASR:
- Replaces RNN encoder and decoder with **multi-head self-attention** layers.
- Fully parallelizable during training.
- Captures long-range dependencies in a single attention pass.

**Challenge unique to audio:** Speech sequences are much **longer** than text. A 10-second utterance produces ~1,000 frames with 10ms stride — far longer than typical NLP sequences. This makes the $O(N^2)$ attention cost a critical bottleneck.

**Solution:** Subsampling / strided convolution layers at the input to reduce sequence length by 4–8× before the Transformer layers.

> **Reference:** Dong et al., *"Speech-Transformer: A No-Recurrence Sequence-to-Sequence Model for Speech Recognition"* (InterSpeech 2018)

---

### Slide 7 — Conformer: The Best of Both Worlds

The **Conformer** (*Gulati et al., 2020*) became the dominant ASR architecture by combining:
- **Convolution** (local feature patterns, like phonetic patterns).
- **Self-attention** (global context, like language-level dependencies).

**Architecture:** Each Conformer block interleaves:
1. Feed-Forward (½) → 2. Multi-Head Self-Attention → 3. **Convolution Module** → 4. Feed-Forward (½) → LayerNorm

The **convolution module** uses depthwise separable convolutions to efficiently capture **local acoustic patterns** (formants, phoneme transitions) that attention alone handles poorly.

**Result:** Conformer achieved **state-of-the-art** on LibriSpeech (2.1% WER on test-clean), surpassing both pure Transformer and pure CNN approaches. It became the go-to architecture for production ASR systems (Google, NVIDIA NeMo).

> **Reference:** Gulati et al., *"Conformer: Convolution-augmented Transformer for Speech Recognition"* (InterSpeech 2020)

---

### Slide 8 — Self-Supervised Speech: wav2vec 2.0 (THE Breakthrough)

**The problem:** Labeled speech data is **expensive** (requires human transcription). Most languages have <100 hours of labeled data. Meanwhile, **unlabeled** audio is plentiful (podcasts, YouTube, radio).

**wav2vec 2.0** (*Baevski et al., 2020*) solved this with a BERT-like self-supervised framework for speech:

1. **CNN Feature Encoder:** Raw waveform → latent speech representations (local features).
2. **Quantization Module:** Discretizes latent representations into a finite codebook (like VQ-VAE), producing **pseudo-tokens**.
3. **Transformer Encoder:** Processes the full sequence with masked positions.
4. **Contrastive Loss:** The model must identify the correct quantized representation for each masked position from a set of distractors.

**Impact:**
- With only **10 minutes** of labeled data + pre-training on 53K hours of unlabeled audio, wav2vec 2.0 achieved **4.8 WER** on LibriSpeech test-clean.
- This is the **"BERT moment"** for speech: pre-train once, fine-tune cheaply on any downstream task.

> **Reference:** Baevski et al., *"wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations"* (NeurIPS 2020)

---

### Slide 9 — HuBERT: Discrete Target Prediction

**HuBERT** (*Hsu et al., 2021*) refined self-supervised speech learning:

- Instead of contrastive learning, HuBERT uses **offline clustering** (k-means on MFCC features) to generate **pseudo-labels** for masked frames.
- The model is trained to predict these discrete targets — like **masked language modeling** but with discovered speech units.
- Iteratively re-clusters using the model's own representations for better targets.

**Advantage over wav2vec 2.0:** Simpler training objective, more stable, and produces representations that transfer well to non-ASR tasks (emotion recognition, speaker ID, etc.).

Both wav2vec 2.0 and HuBERT demonstrated that **self-supervised pre-training** is the most effective way to build speech foundation models — echoing the pre-training revolution in NLP.

> **Reference:** Hsu et al., *"HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units"* (IEEE/ACM TASLP, 2021)

---

### Slide 10 — Whisper: Scaling Supervised ASR to the Limit

**Whisper** (*Radford et al., 2022*) took the opposite approach to self-supervised models: **massive supervised scaling**.

- Trained on **680,000 hours** of labeled audio from the internet (web-scraped transcripts).
- Standard encoder-decoder Transformer (no Conformer tricks).
- **Multi-task format:** A single model handles ASR, translation, language ID, timestamp prediction, and voice activity detection through **special tokens** in the decoder prompt.

| Model | Parameters | English WER (test-clean) |
|-------|-----------|--------------------------|
| Whisper tiny | 39M | 7.6% |
| Whisper base | 74M | 5.0% |
| Whisper small | 244M | 3.4% |
| Whisper medium | 769M | 2.9% |
| Whisper large-v3 | 1.55B | 2.0% |

**Key insight (Song Han perspective):** Whisper shows that brute-force scaling works, but the resulting model is **far too large for edge** — a Whisper large model requires ~3GB of memory and is too slow for real-time on mobile. This motivates all the efficiency techniques we teach in this course.

> **Reference:** Radford et al., *"Robust Speech Recognition via Large-Scale Weak Supervision"* (ICML 2023)

---

### Slide 11 — Audio Spectrogram Transformer (AST): ViT for Audio

**AST** (*Gong et al., 2021*) directly applies the **Vision Transformer (ViT)** to audio classification:

1. Convert audio to a **Mel-spectrogram** (2D image).
2. Split into **16×16 patches** (just like ViT).
3. Flatten and project patches into token embeddings.
4. Process with a standard Transformer encoder.
5. Classify using mean pooling or [CLS] token.

**Key result:** AST achieved **state-of-the-art** on AudioSet (0.485 mAP) and ESC-50 (95.6% accuracy) — **surpassing all CNN-based methods** — by leveraging ImageNet pre-trained ViT weights. This demonstrates the power of **cross-modal transfer** (vision → audio).

**Efficiency concern:** Like ViT (Lecture 16), AST inherits the $O(N^2)$ attention complexity. For a 10-second clip with small patches, the sequence length can be ~1,200 tokens — motivating efficient attention methods.

> **Reference:** Gong et al., *"AST: Audio Spectrogram Transformer"* (InterSpeech 2021)

---

### Slide 12 — Efficiency Technique 1: Architectural Efficiency

Applying course fundamentals to audio Transformers:

**SqueezeFormer** (*Kim et al., 2022*): A macro/micro redesign of Conformer for mobile ASR.
- **Macro:** Temporal U-Net structure — progressively reduces sequence length then upsamples.
- **Micro:** Replaces the Conformer's sandwich structure with a more efficient ordering and uses **depthwise separable convolutions** throughout.
- **Result:** Same WER as Conformer at **significantly fewer MACs** — suitable for mobile deployment.

**Efficient Conformer** (*Burchi & Timofte, 2021*):
- Progressive downsampling of the encoder sequence using **strided attention** — reducing both computation and memory.
- Achieves Conformer-level accuracy with 28% fewer parameters.

> **References:**
> - Kim et al., *"SqueezeFormer: An Efficient Transformer Backbone for Mobile ASR"* (NeurIPS 2022)
> - Burchi & Timofte, *"Efficient Conformer: Progressive Downsampling and Grouped Attention for Automatic Speech Recognition"* (ASRU 2021)

---

### Slide 13 — Efficiency Technique 2: Knowledge Distillation for ASR

**Distil-Whisper** (*Gandhi et al., 2023*): Applies the course's Knowledge Distillation framework (Lecture 9) specifically to Whisper.

- **Teacher:** Whisper large-v2 (1.55B parameters).
- **Student:** 2-layer decoder (vs. 32 in teacher), keeping the full encoder.
- **Training:** Pseudo-labeling on 22K hours of audio using the teacher's transcriptions.
- **Result:** **6× faster** than Whisper large-v2, within 1% WER on out-of-distribution benchmarks.

**Key design choice:** The encoder is more important than the decoder in ASR — the student retains the full encoder but aggressively shrinks the decoder. This is the opposite of LLM distillation where the decoder dominates.

> **Reference:** Gandhi et al., *"Distil-Whisper: Robust Knowledge Distillation via Large-Scale Pseudo Labelling"* (2023)

---

### Slide 14 — Efficiency Technique 3: Quantization and Pruning for ASR

**Quantization:**
- INT8 quantization of Conformer/Whisper models using **Post-Training Quantization (PTQ)** — directly applies techniques from Lectures 5–6.
- **Challenge:** Attention layers and layer normalization have **high dynamic range** activations, similar to LLMs. **SmoothQuant**-style techniques (Lecture 13) transfer directly.
- Whisper INT8 achieves **<0.5% WER degradation** with **2× memory reduction**.

**Pruning:**
- Structured pruning of attention heads and FFN dimensions (Lectures 3–4).
- **Finding:** In Conformer-based ASR, the **convolution modules** are more sensitive to pruning than the attention modules — the opposite of LLMs.

**Combined:** Quantization + pruning can reduce Whisper-medium to **~200MB** (from ~1.5GB) with <1% WER increase — bringing it into mobile deployment range.

> **References:**
> - Fasoli et al., *"Fast Conformer-based End-to-End Speech Recognition with Quantization"* (2021)
> - Peng et al., *"Structured Pruning of Self-Supervised Pre-Trained Models for Speech Recognition and Understanding"* (ICASSP 2023)

---

### Slide 15 — Efficiency Technique 4: Streaming and Real-Time ASR

**The Problem:** Standard Transformer ASR requires the **full utterance** before processing (non-streaming). But real-time applications (voice assistants, live captioning) need **streaming** with minimal latency.

**Solutions:**
- **Chunked Attention:** Process audio in overlapping chunks; each chunk attends only to itself and a limited left context. Achieves near-offline quality with bounded latency.
- **Emformer (Efficient Memory Transformer):** Uses a memory bank mechanism — summarizes past context into compact memory vectors rather than caching all past keys/values (analogous to **StreamingLLM** from Lecture 13).
- **CTC-based streaming:** CTC's conditional independence assumption naturally supports streaming — each frame can be decoded independently with a greedy or beam search.

> **References:**
> - Shi et al., *"Emformer: Efficient Memory Transformer Based Acoustic Model for Low Latency Streaming Speech Recognition"* (ICASSP 2021)
> - An et al., *"Unified Streaming and Non-streaming Conformer for Speech Recognition"* (InterSpeech 2022)

---

### Slide 16 — TinyML for Audio: Keyword Spotting on Microcontrollers

**Keyword Spotting (KWS)** is the ultimate edge audio task — runs on always-on hardware with:
- **<256 KB RAM**, **<1 MB Flash** (Cortex-M class MCU).
- **<1 mW** power budget (battery life: months to years).

**Architecture evolution:**
1. **DS-CNN** (*Hello Edge*, *Zhang et al., 2017*): Depthwise separable CNNs on MFCC features — the baseline for MCU keyword spotting.
2. **Temporal Convolution (TC-ResNet):** 1D convolutions along time axis — even smaller footprint.
3. **MCUNet-style NAS for Audio:** Applying Neural Architecture Search (Lectures 7–8) with hardware-aware constraints to find optimal KWS architectures for specific MCU targets.

| Model | Parameters | MACs | Accuracy (GSC v2) |
|-------|-----------|------|-------------------|
| DS-CNN (L) | 500K | 56M | 95.4% |
| TC-ResNet14 | 305K | 28M | 96.6% |
| NAS-optimized | 63K | 6M | 95.8% |

> **Reference:** Zhang et al., *"Hello Edge: Keyword Spotting on Microcontrollers"* (2017)

---

### Slide 17 — Summary: The Audio Transformer Arc

| Era | Model | Key Innovation | Limitation |
|-----|-------|---------------|------------|
| Pre-DL | HMM-GMM | Statistical sequence modeling | Modular, expert-designed |
| DNN-HMM | DNN + HMM | Neural acoustic model | Still modular |
| End-to-End RNN | Deep Speech 2, LAS | Single model, CTC/attention | Sequential, slow |
| **Transformer** | Speech-Transformer, Conformer | Parallel, global context + local conv | Large, non-streaming |
| **Self-Supervised** | **wav2vec 2.0**, HuBERT | Pre-train on unlabeled audio | Compute-heavy pre-training |
| **Scaled** | Whisper | Brute-force 680K hours | 1.55B params, too large for edge |
| **Efficient** | SqueezeFormer, Distil-Whisper | NAS, KD, quantization, pruning | Active research area |
| **TinyML** | DS-CNN, NAS-KWS | <256KB MCU deployment | Limited vocabulary |

**The pattern mirrors Language exactly:** RNN → Transformer → scale up → compress back down for deployment.

---

## References

- Baevski et al., *"wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations"* (NeurIPS 2020)
- Radford et al., *"Robust Speech Recognition via Large-Scale Weak Supervision"* (ICML 2023)
- Gulati et al., *"Conformer: Convolution-augmented Transformer for Speech Recognition"* (InterSpeech 2020)
- Gong et al., *"AST: Audio Spectrogram Transformer"* (InterSpeech 2021)
- Hsu et al., *"HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units"* (2021)
- Kim et al., *"SqueezeFormer: An Efficient Transformer Backbone for Mobile ASR"* (NeurIPS 2022)
- Gandhi et al., *"Distil-Whisper: Robust Knowledge Distillation via Large-Scale Pseudo Labelling"* (2023)
- Graves et al., *"Connectionist Temporal Classification"* (ICML 2006)
- Chan et al., *"Listen, Attend and Spell"* (ICASSP 2016)
- Amodei et al., *"Deep Speech 2"* (ICML 2016)
- Dong et al., *"Speech-Transformer"* (InterSpeech 2018)
- Hinton et al., *"Deep Neural Networks for Acoustic Modeling in Speech Recognition"* (2012)
- Zhang et al., *"Hello Edge: Keyword Spotting on Microcontrollers"* (2017)
- Shi et al., *"Emformer: Efficient Memory Transformer Based Acoustic Model"* (ICASSP 2021)
- Peng et al., *"Structured Pruning of Self-Supervised Pre-Trained Models for Speech"* (ICASSP 2023)

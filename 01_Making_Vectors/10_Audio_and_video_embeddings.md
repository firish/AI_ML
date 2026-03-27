## Audio and Video Encoders

### The Key Insight

Audio and video don't need new architectures. They get **converted into formats we already know how to encode**:

```text
Audio  → mel spectrogram (a 2D image)  → ViT or CNN (file 08)
Video  → sequence of image frames      → image encoder + temporal aggregation

The hard part isn't the encoder — it's the conversion step.
```

---

## Part 1: Audio Encoders

### Three Approaches to Audio

```text
Approach 1: Audio → Transcribe → Text Embeddings
    "What was said?" (semantic meaning)
    Best for: podcasts, meetings, call centers, voice search

Approach 2: Audio → Spectrogram → Audio Embeddings
    "What does it sound like?" (acoustic similarity)
    Best for: music similarity, sound effects, speaker ID, environmental sounds

Approach 3: CLAP (Audio + Text in same space)
    "Search sounds with text" (cross-modal)
    Best for: text → audio search, zero-shot audio classification
```

---

### Approach 1: Transcribe First, Then Embed the Text

The simplest path. Used when you care about **meaning**, not **sound**.

```text
Audio waveform (someone speaking)
    ↓
Speech-to-Text model (Whisper, Deepgram, AssemblyAI)
    ↓
Transcript: "The quarterly revenue exceeded expectations"
    ↓
Text encoder (Sentence-BERT, OpenAI embeddings — file 06/07)
    ↓
Vector: [0.12, -0.34, 0.56, ...]   (768-d, in text space)
    ↓
Semantic search over meaning
```

**When this is the right choice:**
- Searching spoken content by topic ("find the part where they discuss pricing")
- Indexing podcasts, lectures, customer support calls
- Any case where the audio is just speech carrying language

**When this fails:**
- Music (no words, or words don't capture the sound)
- Sound effects ("thunder", "glass breaking" — no speech to transcribe)
- Speaker identification ("whose voice is this?" — transcript loses speaker identity)
- Emotion/tone detection ("they said 'fine' but sounded angry")

---

### Approach 2: Audio Embeddings Directly

When you care about **how it sounds**, you need to encode the audio signal itself.

#### Step 1: Convert Audio to a Mel Spectrogram

Raw audio is a 1D waveform — just amplitude values over time. This is hard for neural networks to work with directly. The standard trick: convert to a **mel spectrogram**, which is a 2D image.

```text
Raw waveform:
    amplitude
    ▲
    │   ╱╲  ╱╲╱╲    ╱╲
    │  ╱  ╲╱      ╲╱  ╲╱
    │╱
    └──────────────────────→ time
    Just a list of numbers: [0.1, 0.3, -0.2, 0.5, ...]

Mel spectrogram:
    frequency (mel scale)
    ▲
    │ ░░░░▓▓▓▓░░░░░░▓▓▓▓░░
    │ ░░▓▓████▓▓░░▓▓████▓▓
    │ ▓▓██████████████████▓▓
    │ ████████████████████████
    └──────────────────────→ time

    A 2D image where:
    - x-axis = time
    - y-axis = frequency (log scale, matching human hearing)
    - pixel brightness = energy at that frequency at that time

    Typical size: 128 frequency bins × T time frames
    (T depends on audio length — 10 seconds ≈ 1000 frames)
```

**How it's computed:**

```text
1. Chop the waveform into overlapping windows (25ms each, 10ms hop)
2. Apply FFT (Fast Fourier Transform) to each window
   → converts "amplitude over time" to "energy per frequency"
3. Map frequencies to mel scale (compresses high frequencies,
   expands low ones — matches how human ears perceive pitch)
4. Take the log of the energies (humans perceive loudness logarithmically)

Result: a 2D matrix that looks like a greyscale image.
```

Why mel scale? Humans hear the difference between 200Hz and 400Hz easily, but can barely tell 8000Hz from 8200Hz. The mel scale reflects this — it spaces low frequencies apart and compresses high frequencies.

#### Step 2: Encode the Spectrogram

Once you have a 2D "image", you can use the same architectures from file 08:

**Option A: CNN-based (PANNs — Pre-trained Audio Neural Networks)**

```text
Mel spectrogram (128 × T)
    ↓
Treat as a 1-channel grayscale image
    ↓
CNN (ResNet or similar) — same conv → ReLU → pool pipeline as file 08
    ↓
Global Average Pooling
    ↓
Audio vector (2048-d)

PANNs were trained on AudioSet (2M audio clips, 527 sound classes).
Same idea as training ResNet on ImageNet — supervised classification,
then throw away the classification head and keep the embeddings.
```

**Option B: Transformer-based (AST — Audio Spectrogram Transformer)**

```text
Mel spectrogram (128 × T)
    ↓
Chop into patches (16 × 16) — exactly like ViT chops images into patches
    ↓
Flatten each patch → linear projection → patch embeddings
    ↓
Add [CLS] token + position embeddings
    ↓
12 transformer blocks (identical to ViT / BERT — same attention, same FFN)
    ↓
Take [CLS] output
    ↓
Audio vector (768-d)

AST is literally ViT applied to spectrograms.
The transformer doesn't know it's looking at audio — it just sees patches.
Often initialised from a pre-trained ViT (ImageNet weights transfer
surprisingly well to spectrograms).
```

**Option C: HuBERT / wav2vec 2.0 (Waveform directly)**

```text
Raw waveform (no spectrogram conversion)
    ↓
CNN feature extractor (learns its own "spectrogram-like" representation)
    ↓
Transformer blocks
    ↓
Audio vector

Training game: Masked prediction (like BERT's MLM)
    - Mask portions of the audio
    - Predict the masked parts from context
    - Self-supervised — no labels needed

HuBERT = "Hidden-Unit BERT" — BERT's masked prediction game, but for audio.

Best for: tasks where speech nuance matters (speaker ID, emotion,
         language identification) — captures more than a spectrogram.
```

---

### Approach 3: CLAP — The CLIP of Audio

Same contrastive alignment idea as CLIP (file 09), but for audio + text:

```text
┌─────────────────────┐         ┌─────────────────────┐
│    Audio Encoder     │         │    Text Encoder      │
│                      │         │                      │
│  Spectrogram → CNN   │         │  Transformer         │
│  or AST              │         │  (RoBERTa or similar)│
│                      │         │                      │
│  Input: audio clip   │         │  Input: text caption  │
│  Output: 512-d vec   │         │  Output: 512-d vec    │
└──────────┬──────────┘         └──────────┬──────────┘
           │                                │
           └──────── SAME space ───────────┘
```

**Training data:** (audio clip, text description) pairs

```text
(sound_of_thunder.wav,       "Thunder rumbling during a storm")
(dog_barking.wav,            "A dog barking loudly")
(piano_melody.wav,           "Soft piano playing a gentle melody")
```

**Training game:** Identical to CLIP — contrastive loss on the similarity matrix, both directions.

```text
                        T₀(thunder)  T₁(dog)    T₂(piano)
A₀(thunder audio)    [   0.94        0.08       0.05   ]
A₁(dog bark audio)   [   0.06        0.91       0.03   ]
A₂(piano audio)      [   0.04        0.07       0.93   ]

Loss: push diagonal up, off-diagonal down (same as CLIP).
```

**What you can do after training:**

```text
Text → Audio search:
    "birds chirping in a forest" → find matching audio clips

Audio → Text search:
    Upload unknown sound → get text description

Zero-shot audio classification:
    Labels: ["gunshot", "firework", "door slam"]
    Encode each as text, compare with audio vector
    → classify without training a dedicated classifier

Audio → Audio similarity (within CLAP's space):
    Find sounds similar to a given clip
```

**Key models:**

```text
CLAP (LAION, 2023):
    Audio encoder: HTS-AT (Hierarchical Token-Semantic Audio Transformer)
    Text encoder: RoBERTa
    Trained on: AudioSet + other datasets (~630K audio-text pairs)
    Open-source, most widely used

Microsoft CLAP (2022):
    Earlier version, CNN-based audio encoder
    Smaller training set
```

---

### Audio Models: Quick Comparison

```text
| Model        | Input        | Architecture      | Dims | Training          | Best for                      |
| ------------ | ------------ | ----------------- | ---- | ----------------- | ----------------------------- |
| Whisper+SBERT| Speech audio | STT → text encoder| 768  | Two-stage pipeline| Spoken content search         |
| PANNs        | Spectrogram  | CNN (ResNet-like) | 2048 | Supervised (AudioSet)| Sound classification, simple |
| AST          | Spectrogram  | ViT on patches    | 768  | Supervised (AudioSet)| Sound classification, better |
| HuBERT       | Raw waveform | CNN + Transformer | 768  | Self-supervised (MLM)| Speaker ID, emotion, speech  |
| wav2vec 2.0  | Raw waveform | CNN + Transformer | 768  | Self-supervised (MLM)| Speech recognition backbone  |
| CLAP         | Audio + text | HTS-AT + RoBERTa  | 512  | Contrastive pairs | Text↔audio search, zero-shot  |
```

---

## Part 2: Video Encoders

### The Challenge

Video = sequence of images + audio. Two extra problems vs single images:

```text
1. Temporal understanding:
    A single frame of someone throwing a ball looks identical
    to someone catching a ball. You need multiple frames to
    understand the ACTION.

2. Scale:
    A 10-second video at 30fps = 300 frames.
    Encoding every frame independently with ViT = 300 forward passes.
    Too expensive for real-time use.
```

---

### Approach 1: Sample Frames → Image Encoder → Aggregate

The simplest and most common approach:

```text
Video (300 frames)
    ↓
Sample: pick N evenly-spaced frames (N = 8 or 16 typically)
    ↓
Image encoder (ViT, DINOv2, CLIP — from file 08/09):
    Frame 0  → v₀  (768-d)
    Frame 4  → v₁  (768-d)
    Frame 8  → v₂  (768-d)
    ...
    Frame 28 → v₇  (768-d)
    ↓
Aggregate:
    Option A: Mean pool → average all frame vectors → 1 video vector (768-d)
    Option B: Temporal transformer → attend across frames → [CLS] → 1 video vector
    ↓
Video embedding (768-d)
```

**Mean pooling** works surprisingly well for "what is this video about?" retrieval. It loses temporal order (can't tell throwing from catching) but captures the overall scene content.

**Temporal transformer** adds a small transformer on top that attends across frame vectors — this captures ordering and motion. Used when actions matter.

---

### Approach 2: Video Transformers (ViViT, TimeSformer)

Extend ViT to handle video natively:

```text
ViViT (Video Vision Transformer):

Video: T frames × 224 × 224
    ↓
Create "tubelet" patches:
    Instead of 2D patches (16×16 from one frame),
    create 3D patches (2 frames × 16 × 16 = spatiotemporal tubes)
    ↓
Flatten + project each tubelet → one vector
    ↓
Add position embeddings (spatial + temporal)
    ↓
Transformer blocks with attention across ALL tubelets
    (a patch from frame 1 can attend to a patch from frame 8)
    ↓
Pooling → video vector

The transformer sees space AND time simultaneously.
```

**TimeSformer** uses a factored approach to make this tractable:

```text
Instead of full spatiotemporal attention (every patch attends to
every patch across all frames — extremely expensive):

TimeSformer alternates:
    1. Spatial attention: each patch attends to other patches in the SAME frame
    2. Temporal attention: each patch attends to the SAME position in OTHER frames

Much cheaper than full attention, nearly as accurate.
```

---

### Approach 3: Video + Text (VideoCLIP, X-CLIP)

Same CLIP pattern, but for video:

```text
┌─────────────────────┐         ┌─────────────────────┐
│   Video Encoder      │         │    Text Encoder      │
│                      │         │                      │
│  Frames → ViT →     │         │  Transformer         │
│  temporal pooling    │         │                      │
│                      │         │                      │
│  Input: video clip   │         │  Input: text caption  │
│  Output: 512-d vec   │         │  Output: 512-d vec    │
└──────────┬──────────┘         └──────────┬──────────┘
           │                                │
           └──────── SAME space ───────────┘

Training: contrastive loss on (video clip, caption) pairs
    Same game as CLIP — push matching pairs close, non-matching far.

Enables:
    "Find videos of someone doing a backflip" → text query → video results
    Upload video → get text description
    Zero-shot action classification
```

---

### Video Models: Quick Comparison

```text
| Model       | Approach                    | Best for                          |
| ----------- | --------------------------- | --------------------------------- |
| CLIP+pool   | Sample frames → CLIP → mean | Simple video search (by content)  |
| ViViT       | Spatiotemporal patches      | Action recognition                |
| TimeSformer | Factored space/time attn    | Action recognition (more efficient)|
| VideoMAE    | Masked video modelling      | Self-supervised video features    |
| X-CLIP      | Video + text contrastive    | Text → video search               |
| InternVideo | Large-scale video foundation| Best general video understanding  |
```

---

## The Pattern Across All Modalities

```text
Modality     Raw input              Conversion              Encoder
─────────────────────────────────────────────────────────────────────
Text         "The cat sat"          Tokenize → IDs          Transformer (BERT)
Image        224×224×3 pixels       Patch + flatten          ViT or CNN
Audio        Waveform samples       Mel spectrogram          ViT on spectrogram (AST)
                                    (or raw → CNN)           (or HuBERT on waveform)
Video        Sequence of frames     Sample + patch           ViT + temporal attention

Cross-modal alignment (CLIP pattern):
    Text ↔ Image:  CLIP, SigLIP
    Text ↔ Audio:  CLAP
    Text ↔ Video:  X-CLIP, VideoCLIP
    Everything:    ImageBind (Meta) — 6 modalities, 1 shared space

The building blocks are always the same:
    1. Convert raw input → sequence of vectors
    2. Transformer blocks (attention → residual → LayerNorm → FFN)
    3. Pooling → one vector
    4. (Optional) Contrastive training to align with another modality
```

---

## How to Choose for Audio

```text
"I have speech and want to search by topic"
    → Whisper transcription → text encoder (don't bother with audio embeddings)

"I want to find similar-sounding audio clips"
    → PANNs or AST (audio embeddings from spectrogram)

"I want to search audio with text queries"
    → CLAP (audio + text in same space)

"I need speaker identification or emotion detection"
    → HuBERT or wav2vec 2.0 (captures voice characteristics)
```

## How to Choose for Video

```text
"I want to search videos by content/scene"
    → CLIP on sampled frames + mean pool (simplest, works well)

"I need action recognition (throwing vs catching)"
    → ViViT or TimeSformer (temporal awareness)

"I want text → video search"
    → X-CLIP or InternVideo

"I'm prototyping"
    → DINOv2 or CLIP on sampled frames → mean pool
      (reuse image encoder, no video-specific model needed)
```

## Multimodal Encoders (CLIP and Variants)

### The Problem

So far, text and images live in **completely separate worlds**:

```text
Text encoder:   "A golden retriever playing in grass"  → [0.12, -0.34, ...]   (text space)
Image encoder:  photo_of_golden_retriever.jpg          → [0.87, 0.23, ...]    (image space)

These vectors are in DIFFERENT spaces.
You cannot compare them — cosine similarity between them is meaningless.
```

This means:
- You can search text with text (file 06)
- You can search images with images (file 08)
- But you **cannot** search images with text, or text with images

Real-world tasks that need cross-modal search:
- "Find me photos of red running shoes" → search an image database with a text query
- Upload a photo → "What is this?" → find matching text descriptions
- Zero-shot classification: classify images using text labels the model has never been trained on

### The Solution

**Train two encoders (one for text, one for images) so they output vectors in the SAME shared space.**

```text
Before CLIP (separate spaces):
    "dog playing"     → [0.12, -0.34]     (text space — can't compare with image vectors)
    photo_of_dog.jpg  → [0.87,  0.23]     (image space — can't compare with text vectors)

After CLIP (shared space):
    "dog playing"     → [0.85,  0.21]  ─┐
                                         ├── cosine similarity = 0.95 ✓
    photo_of_dog.jpg  → [0.83,  0.24]  ─┘

    "car on highway"  → [-0.3,  0.71]  ─┐
                                         ├── cosine similarity = 0.08 ✗
    photo_of_dog.jpg  → [0.83,  0.24]  ─┘
```

The key insight: you don't need a new architecture. You take an **existing text encoder** and an **existing image encoder**, and train them **together** with a contrastive loss that aligns their output spaces.

---

## CLIP: The Base Model (OpenAI, 2021)

### Architecture

Two completely separate encoders that share nothing except the training signal:

```text
┌─────────────────────┐         ┌─────────────────────┐
│    Image Encoder     │         │    Text Encoder      │
│                      │         │                      │
│  ViT-L/14 or        │         │  Transformer         │
│  ResNet-50           │         │  (12 layers, 512-d)  │
│                      │         │                      │
│  Input: 224×224 img  │         │  Input: text tokens   │
│  Output: 768-d vec   │         │  Output: 768-d vec    │
└──────────┬──────────┘         └──────────┬──────────┘
           │                                │
           ▼                                ▼
     v_image (768-d)                  v_text (768-d)
           │                                │
           └──────── SAME space ───────────┘

Both vectors are L2-normalized to unit length.
Now: cosine_similarity(v_image, v_text) is meaningful.
```

The image encoder is a standard ViT or ResNet (file 08). The text encoder is a standard transformer (file 06). CLIP doesn't invent new architectures — it invents a new **training game** that forces them into the same space.

---

### Training Data

```text
400 million (image, text) pairs scraped from the internet.
Sources: alt-text on web pages, captions, titles.

Examples:
    (photo_of_retriever.jpg,     "A happy golden retriever playing fetch")
    (photo_of_eiffel_tower.jpg,  "The Eiffel Tower at sunset in Paris")
    (photo_of_pizza.jpg,         "Delicious margherita pizza on a plate")

No human labelling needed — the text already exists on the web.
```

---

### The Training Game: Contrastive Learning on Pairs

This is the core innovation. Walk through it with a toy batch of 3 pairs:

**Step 1: Assemble a mini-batch of B pairs**

```text
Pair 0:  (I₀ = dog photo,    T₀ = "A happy dog playing")
Pair 1:  (I₁ = tower photo,  T₁ = "The Eiffel Tower at sunset")
Pair 2:  (I₂ = pizza photo,  T₂ = "Margherita pizza on a plate")
```

**Step 2: Encode everything**

```text
Image encoder:     I₀ → v_I₀     I₁ → v_I₁     I₂ → v_I₂
Text encoder:      T₀ → v_T₀     T₁ → v_T₁     T₂ → v_T₂
```

**Step 3: L2-normalize all vectors to unit length**

```text
v = v / ||v||

Example: v = (4, -3, 12)
||v|| = √(4² + (-3)² + 12²) = √169 = 13
v_normalized = (4/13, -3/13, 12/13) ≈ (0.308, -0.231, 0.923)

After normalization, cosine similarity = simple dot product.
```

**Step 4: Compute the similarity matrix**

Every image vector dotted with every text vector:

```text
S = v_I · v_Tᵀ       (B × B matrix)

            T₀(dog)   T₁(tower)  T₂(pizza)
I₀(dog)   [  0.92      0.10       0.05  ]
I₁(tower) [  0.12      0.88       0.07  ]
I₂(pizza) [  0.05      0.09       0.95  ]

Diagonal = correct pairs (should be HIGH)
Off-diagonal = wrong pairs (should be LOW)
```

**Step 5: Apply contrastive loss (both directions)**

Treat each row as a classification problem: "which text matches this image?"
Treat each column as a classification problem: "which image matches this text?"

```text
Row loss (image → text):
    Row 0: softmax([0.92, 0.10, 0.05]) → target = index 0 → cross-entropy loss
    Row 1: softmax([0.12, 0.88, 0.07]) → target = index 1 → cross-entropy loss
    Row 2: softmax([0.05, 0.09, 0.95]) → target = index 2 → cross-entropy loss

Column loss (text → image):
    Col 0: softmax([0.92, 0.12, 0.05]) → target = index 0 → cross-entropy loss
    Col 1: softmax([0.10, 0.88, 0.09]) → target = index 1 → cross-entropy loss
    Col 2: softmax([0.05, 0.07, 0.95]) → target = index 2 → cross-entropy loss

Total loss = (mean of row losses + mean of column losses) / 2
```

Why both directions? "Image finds its text" AND "text finds its image" — symmetric alignment.

**Step 6: Backprop through BOTH encoders**

```text
Loss
 ├── gradient flows into image encoder → updates image encoder weights
 └── gradient flows into text encoder  → updates text encoder weights

Both encoders adjust so that:
    - Matched pairs' vectors move closer (diagonal goes up)
    - Unmatched pairs' vectors move apart (off-diagonal goes down)
```

**Step 7: After training (400M pairs, many epochs)**

Both encoders output vectors in the same space. A photo of a dog and the text "a cute puppy" land near each other — even though one went through a ViT and the other through a text transformer.

---

### Temperature Parameter (τ)

The actual similarity matrix uses a **learned temperature** to sharpen the distribution:

```text
S[i,j] = cos(v_Iᵢ, v_Tⱼ) / τ

τ starts at ~0.07 and is learned during training.

With τ = 1.0:    softmax([0.92, 0.10, 0.05]) = [0.52, 0.23, 0.22]  (soft, unclear)
With τ = 0.07:   softmax([13.1, 1.43, 0.71]) = [0.99, 0.01, 0.00]  (sharp, decisive)

Low temperature → the model must be very confident about the correct pair.
This pushes training harder and produces better features.
```

---

### What Can You Do With a Trained CLIP?

**1. Text → Image Search**

```text
query = "red running shoes"
v_query = text_encoder(query)       → 768-d vector (in shared space)
results = nearest_neighbors(v_query, image_vector_db)
→ returns photos of red running shoes, ranked by similarity
```

**2. Image → Text Search**

```text
photo = "mystery_object.jpg"
v_photo = image_encoder(photo)      → 768-d vector (in shared space)
results = nearest_neighbors(v_photo, text_vector_db)
→ returns text descriptions that match the photo
```

**3. Zero-Shot Image Classification**

Classify images into categories the model was NEVER explicitly trained on:

```text
Labels: ["cat", "dog", "car", "airplane"]
Prompts: ["a photo of a cat", "a photo of a dog", "a photo of a car", "a photo of an airplane"]

v_prompts = [text_encoder(p) for p in prompts]       → 4 text vectors
v_image = image_encoder("mystery_photo.jpg")          → 1 image vector

similarities = [cos(v_image, v_p) for v_p in v_prompts]
    = [0.91, 0.15, 0.08, 0.03]

Prediction: "cat" (highest similarity)

This works because CLIP learned the CONCEPT of "cat" from millions of
(cat photo, "a cat") pairs — it doesn't need a dedicated cat classifier.
```

**4. Multimodal RAG**

```text
Store both text chunks AND images in the same vector DB (using their respective encoders).
Query with text or image → retrieve relevant content of EITHER type.
Feed results into an LLM for generation.
```

---

## CLIP Variants

CLIP established the paradigm. Everything after it tweaks the architecture, training data, or loss function — the core idea (two encoders, contrastive alignment) stays the same.

---

### OpenCLIP (2022) — Open-Source CLIP on More Data

```text
What it is:
    Open-source reproduction of CLIP by LAION.

What changed:
    Architecture: same (ViT + text transformer)
    Training game: same (contrastive on pairs)
    Training data: LAION-2B (2 BILLION image-text pairs vs CLIP's 400M)
                   5× more data, publicly available

Result:
    Matches or exceeds OpenAI CLIP on most benchmarks.
    Fully open-source (weights, data, code).

When to use:
    Default choice when you want CLIP-style embeddings.
    Best open-source option for text↔image search.
```

---

### SigLIP (Google, 2023) — Better Loss Function

```text
What it is:
    CLIP with a different loss function.

The problem with CLIP's loss:
    CLIP uses softmax over the full B × B matrix.
    Softmax requires a global normalization across all pairs in the batch:
        softmax(row_i) = exp(S[i,j]) / Σ_k exp(S[i,k])

    This "Σ_k" means every GPU must share its similarity scores with every
    other GPU → communication bottleneck → hard to scale batch size.

SigLIP's fix:
    Replace softmax with sigmoid — treat each cell independently:
        loss(i,j) = -log(σ(S[i,j]))      if i == j  (matched pair)
        loss(i,j) = -log(1 - σ(S[i,j]))  if i ≠ j  (unmatched pair)

    Each cell is a binary yes/no: "does this image match this text?"
    No global normalization needed → GPUs don't need to communicate.

    σ = sigmoid:    σ(x) = 1 / (1 + exp(-x))
        σ(0.92 / τ) ≈ 0.99    → matched pair → -log(0.99) ≈ 0.01  (low loss ✓)
        σ(0.10 / τ) ≈ 0.19    → unmatched   → -log(0.81) ≈ 0.21  (low loss ✓)

Result:
    Scales to massive batch sizes (32K+) across many GPUs.
    Slightly better accuracy than CLIP at same compute budget.
    Used by Google's PaLI and Gemini multimodal models.

When to use:
    When you need the best accuracy from a CLIP-style model.
    SigLIP ViT-SO400M is one of the strongest open models available.
```

---

### ALIGN (Google, 2021) — Noisy Data, No Curation

```text
What it is:
    Same architecture and training game as CLIP.

What changed:
    CLIP: 400M pairs, carefully filtered and curated.
    ALIGN: 1.8B pairs, raw alt-text from the web, minimal filtering.

    ALIGN's bet: throw enough noisy data at it and the model
    will learn to ignore the noise. It worked.

Architecture difference:
    Image encoder: EfficientNet (CNN) instead of ViT.
    Text encoder: BERT instead of a custom transformer.

Result:
    Comparable to CLIP despite much noisier data.
    Showed that dataset scale can compensate for noise.

When to use:
    Mostly a research contribution — showed data scale > data quality.
    Not widely used directly (weights were not initially released).
```

---

### BLIP-2 (Salesforce, 2023) — Bridge to Language Models

```text
What it is:
    Connects a frozen image encoder to a frozen LLM via a small trainable bridge.

The problem it solves:
    CLIP outputs a single vector per image — good for search, but can't
    generate detailed descriptions or answer questions about images.
    You want the image understanding of CLIP combined with the language
    generation of an LLM (like GPT).

Architecture:
    ┌──────────────┐         ┌──────────┐         ┌──────────────┐
    │ Frozen Image │  →───→  │ Q-Former │  →───→  │  Frozen LLM  │
    │ Encoder      │         │ (small,  │         │  (generates  │
    │ (e.g., ViT-G)│         │ trainable)│        │   text)      │
    └──────────────┘         └──────────┘         └──────────────┘

    Q-Former = "Querying Transformer"
        A small transformer (188M params) with learnable query tokens.
        Queries attend to the image encoder's output → extract the most
        relevant visual information → feed it to the LLM as "soft prompts."

Training (3 stages):
    Stage 1: Image-text contrastive (like CLIP) — align Q-Former with image encoder
    Stage 2: Image-grounded text generation — Q-Former learns to feed
             useful visual info to a text decoder
    Stage 3: Connect to frozen LLM — Q-Former learns to "translate"
             visual features into tokens the LLM can understand

The key insight:
    Don't retrain the image encoder (expensive, already good).
    Don't retrain the LLM (expensive, already good).
    Just train a small bridge (188M params) to connect them.

When to use:
    Visual question answering ("What's in this image?")
    Image captioning
    Visual chat — this is the architecture behind many multimodal chatbots.
    NOT for simple vector search (use CLIP/SigLIP for that).
```

---

### EVA-CLIP (2023) — Better Initialization

```text
What it is:
    CLIP with a better-initialized image encoder.

The idea:
    Standard CLIP: initialise ViT randomly, then train with contrastive loss.
    EVA-CLIP: pre-train the ViT with masked image modelling (EVA, like MAE)
              THEN train with contrastive loss.

    Better starting point → converges faster → better final accuracy.

Result:
    EVA-02-CLIP-E achieves best open-source zero-shot ImageNet accuracy.
    Stronger visual features than standard CLIP at same model size.

When to use:
    When you want the best possible CLIP-style model and have GPU budget.
```

---

## Summary: The Family Tree

```text
CLIP (2021, OpenAI)
│   Two encoders + contrastive loss on 400M pairs
│   The foundational model that started it all
│
├── OpenCLIP (2022, LAION)
│       Same everything, trained on 2B pairs (open-source)
│
├── SigLIP (2023, Google)
│       Sigmoid loss instead of softmax → scales better, slightly better accuracy
│
├── ALIGN (2021, Google)
│       1.8B noisy pairs, EfficientNet + BERT, proved scale > curation
│
├── EVA-CLIP (2023)
│       Better ViT init via masked image modelling → stronger features
│
└── BLIP-2 (2023, Salesforce)
        Adds a Q-Former bridge to connect image encoder → LLM
        Enables generation (captioning, VQA), not just search
```

---

## How to Choose

```text
"I want text ↔ image search"
    → SigLIP ViT-SO400M (best accuracy, open-source)
    → or OpenCLIP ViT-L/14 (well-tested, widely supported)

"I want zero-shot image classification"
    → SigLIP or EVA-CLIP (highest zero-shot accuracy)

"I want to build a visual chatbot / VQA system"
    → BLIP-2 (bridges image understanding to language generation)
    → This is the precursor to GPT-4V, Gemini Vision, Claude Vision

"I want a simple, well-documented starting point"
    → OpenCLIP ViT-B/32 (smallest, fastest, good enough for prototyping)

"I need an API, don't want to host models"
    → OpenAI CLIP embeddings via API
    → Cohere multimodal embed
    → Voyage multimodal
```

---

## What's Next

CLIP and its variants handle **text + images**. But the same contrastive alignment idea extends further:

```text
Text + Image:     CLIP, SigLIP       (this file)
Text + Audio:     CLAP                (contrastive language-audio pretraining)
Text + Video:     VideoCLIP, X-CLIP   (frame sequences + text)
Text + 3D:        ULIP                (point clouds + text)
Everything:       ImageBind (Meta)    (binds 6 modalities into one space)

The pattern is always the same:
    Two encoders → contrastive loss on paired data → shared vector space
```

These are niche for now and don't warrant their own files until we need them.

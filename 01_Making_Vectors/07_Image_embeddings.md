## Image Encoders

### What Is an Image Encoder?

Same idea as a text encoder: **image in → one fixed-length vector out.**

```text
Text encoder:    "The cat sat on the mat"  → [0.12, -0.34, 0.56, ...]   (768-d)
Image encoder:   photo_of_cat.jpg          → [0.08, 0.91, -0.23, ...]   (768-d or 2048-d)
```

Two architectures evolved for this job:

```text
1. CNN-based (ResNet, 2015)
    Slides small filters across the image
    Builds patterns hierarchically: edges → textures → parts → objects
    Output: 2048-d vector

2. Transformer-based (ViT, 2020)
    Chops image into patches, treats each patch like a "word"
    Runs through the SAME transformer blocks as a text encoder
    Output: 768-d vector
```

Both produce a fixed-length vector. The difference is how they get there.

---

## Part 1: CNN Architecture (ResNet)

### The Pipeline

```text
Image (3 × 224 × 224)
    ↓
Convolution layers: slide small filters → detect edges, textures, parts, objects
    Each step: Convolution → ReLU → (optional) Pool
    Channels grow:         3 → 64 → 256 → 512 → 1024 → 2048
    Spatial dims shrink: 224 → 112 → 56 → 28 → 14 → 7
    ↓
Feature maps (2048 × 7 × 7)
    2048 channels, each a 7×7 grid saying
    "how strongly feature X appears at each location"
    ↓
Global Average Pooling: average each 7×7 grid → one number per channel
    ↓
Vector (2048-d)
    ↓
Normalize to unit length
    ↓
Final embedding
```

Let's understand each building block.

---

### Building Block 1: Convolution

Two terms to know:

| Word | What it is | Think of it as... |
| --- | --- | --- |
| **Channel** | A stacked page that holds one kind of information for every pixel. In a colour photo, the red, green, and blue pages are the first 3 channels. | A separate "layer of ink" in the same image. |
| **Filter** (kernel) | A tiny patch of numbers (e.g., 3×3) that slides over the image. At each position it multiplies its numbers with the pixels underneath and sums → one output number. | A cookie-cutter that looks for a specific pattern (edge, dot, texture). |

#### Toy Example

```text
Input image I (4×4, grayscale — 1 channel):
[[1, 2, 0, 1],
 [0, 1, 3, 2],
 [2, 1, 0, 0],
 [1, 2, 1, 3]]
```

Two filters, each 2×2 (stride = 1, no padding):

```text
Filter A (2×2)               Filter B (2×2)
[[ 1,  0],                   [[ 0, -1],
 [ 0,  1]]                    [ 1,  0]]
```

Each filter slides over every 2×2 patch and computes a dot product:

```text
Filter A at position (0,0):    1×1 + 0×2 + 0×0 + 1×1 = 2
Filter A at position (0,1):    1×2 + 0×0 + 0×1 + 1×3 = 5
... and so on for all 9 positions (3×3 output)
```

The output formula: `h_out = floor((h - f + 2×padding) / stride) + 1`
Here: (4 - 2 + 0) / 1 + 1 = 3. So each filter produces a 3×3 output.

```text
Output — 2 channels × 3 × 3:

Channel A (filter A):          Channel B (filter B):
[[2, 4, 1],                   [[ 0, -1, -1],
 [1, 4, 4],                    [ 3,  0,  2],
 [3, 2, 1]]                    [ 1,  0, -2]]
```

**Key point:** channels jumped from 1 → 2 because we used 2 filters. Each filter creates one output channel. More filters → more pattern types detected.

#### How It Works for Colour Images

```text
Input: 3 × 224 × 224 (one page each for Red, Green, Blue)
Filter: 7 × 7 × 3 — seven-by-seven patch, THREE layers deep
                     (depth always matches input channels)

At every (y, x) position:
    Multiply the filter's 7×7×3 = 147 numbers
    with the 7×7×3 block of pixels underneath
    Sum them all + add a bias → one output number at (y, x)

64 such filters → 64 output channels
Each channel responds to a different pattern (horizontal edge, vertical edge, blob, etc.)
```

---

### Building Block 2: ReLU Activation

After each convolution, apply ReLU to keep strong matches, erase weak/negative ones:

```text
ReLU(x) = max(0, x)

Channel A (unchanged,         Channel B (negatives zeroed):
 no negatives):
[[2, 4, 1],                   [[0, 0, 0],
 [1, 4, 4],                    [3, 0, 2],
 [3, 2, 1]]                    [1, 0, 0]]
```

This introduces non-linearity — without it, stacking convolutions would just be one big linear operation (a matrix multiply), and the network couldn't learn complex patterns.

---

### Building Block 3: Spatial Downsampling (Pooling / Stride)

Shrink the spatial dimensions so the next layer's filters "see" a bigger area of the original image:

```text
Max-pool (2×2, stride 2): take the largest number in each 2×2 window
Average-pool (2×2, stride 2): take the mean of each 2×2 window

Either way: height and width halve.

Channel A after ReLU:          After 2×2 max-pool:
[[2, 4, 1],                   [[4, 4],
 [1, 4, 4],        →           [3, 1]]
 [3, 2, 1]]
    (3×3)                        (2×2, but note: 3×3 with stride 2 gives 1 leftover row/col
                                  — in practice padding handles this)
```

**Why this matters:** After pooling, the next conv layer's 3×3 filter covers a larger region of the original image. This is how CNNs build up from local patterns (edges) to global patterns (objects) — by repeatedly zooming out.

---

### Building Block 4: Global Average Pooling (GAP)

After all the conv blocks, we have a 3D tensor. GAP collapses it to a 1D vector:

```text
Final feature maps after all conv blocks: 2048 × 7 × 7
    → 2048 channels, each a 7×7 grid

GAP: average each 7×7 grid down to one number

Toy example (2 channels):
A_small = [[3, 2],          B_small = [[0, 1],
           [2, 3]]                     [1, 0]]

v_A = (3 + 2 + 2 + 3) / 4 = 2.5
v_B = (0 + 1 + 1 + 0) / 4 = 0.5

Final vector = [2.5, 0.5]

Real ResNet: 2048 channels → 2048-d vector
```

**What GAP does conceptually:** throws away *where* each pattern appeared, keeps *how much* each pattern appeared. The result is a fixed-size summary of the image regardless of input resolution.

---

### Putting It All Together: One Convolution Block Summary

```text
| Stage           | Plain-English job                                                            |
| --------------- | ---------------------------------------------------------------------------- |
| Convolution     | Look for small patterns (edges, blobs). More filters → more pattern types.   |
| ReLU            | Keep strong matches, erase weak/negative ones.                               |
| Stride 2 / Pool | Zoom out: next filters see bigger shapes.                                   |
| GAP (final)     | Throw away where, keep how much → fixed-size vector.                        |
```

---

### ResNet-50: Full Architecture

```text
| Stage                     | Operation          | Filters              | Stride          | Output size (C × H × W)  |
| ------------------------- | ------------------ | -------------------- | --------------- | ------------------------- |
| Input                     | —                  | —                    | —               | 3 × 224 × 224             |
| Conv 1                    | 7 × 7 conv         | 64                   | 2 (+ pad 3)    | 64 × 112 × 112            |
| Max-pool                  | 3 × 3              | —                    | 2 (+ pad 1)    | 64 × 56 × 56              |
| Conv 2 block              | 3 bottleneck layers| 64 → 64 → 256        | 1               | 256 × 56 × 56             |
| Conv 3 block              | 4 bottleneck layers| 128 → 128 → 512      | 2 (first layer) | 512 × 28 × 28             |
| Conv 4 block              | 6 bottleneck layers| 256 → 256 → 1024     | 2               | 1024 × 14 × 14            |
| Conv 5 block              | 3 bottleneck layers| 512 → 512 → 2048     | 2               | 2048 × 7 × 7              |
| GAP (Global-Avg-Pool)     | average over 7 × 7 | —                    | —               | 2048 × 1 × 1              |
| Flatten / L2-norm         | —                  | —                    | —               | 2048-d vector              |
```

Reading this in plain language:
- **Conv 1** — 64 edge detectors scan the RGB image. Output: 64 grey pages, each 112×112.
- **Max-pool** — zoom out. Each page halves in width and height.
- **Conv 2 block** — tiny filters combine edges into corners/texture patches. 256 pattern pages now.
- **Conv 3/4/5 blocks** — keep repeating: zoom out, add more pattern pages. By the last block each "pixel" covers a big chunk of the original photo, and we have 2048 different pattern types.
- **GAP** — squeeze every 7×7 page into one number → a tidy 2048-number list.
- That list is your embedding vector.

---

### Residual (Skip) Connections — What Makes ResNet "Res"Net

**The problem:** Very deep CNNs (20+ layers) actually perform *worse* than shallower ones — not because of overfitting, but because gradients vanish during training.

**The fix:** Instead of learning the full output H(x), each block learns the *residual* F(x) = H(x) - x, then adds the input back:

```text
output = F(x) + x       ← the skip connection

Visually:
input x ──────────────────────┐
   │                          │  (skip / shortcut)
   ▼                          │
[Conv → BN → ReLU → Conv → BN]  │
   │                          │
   ▼                          │
   + ◄────────────────────────┘
   │
  ReLU
   │
   ▼
output
```

**Why it works (two reasons):**

```text
Forward pass:
    output = original info + new info from conv layers
    Even if the conv layers learn nothing useful, the original signal passes through.
    Worst case: F(x) = 0, and the block is a harmless pass-through.

Backward pass:
    Gradient of (x + F(x)) w.r.t. x = 1 + F'(x)
    That "1" means gradients always have a direct path that doesn't shrink.
    Without skip connections: gradients pass through 50 conv layers → vanish.
    With skip connections: gradients shortcut through the "+" → reach early layers.
```

**Dimension mismatch:** When a block changes channels (e.g., 256 → 512), the skip connection uses a 1×1 convolution to match dimensions before adding.

---

### CNN's Fundamental Limitation

Each filter only sees a **small local patch** (3×3 or 7×7 pixels). To understand relationships between distant parts of the image (e.g., "this person's hand is holding that object"), you need many layers stacked so the receptive field gradually grows.

This is the same problem that motivated attention in text — the word "bank" needs to see the whole sentence to know if it means riverbank or financial bank. For images, a cat's ear needs to "know about" the cat's tail to understand "this is a cat."

---

## Part 2: Vision Transformer (ViT)

### The Key Idea

What if we reuse the transformer architecture from text encoders?

**Problem:** Transformers expect a sequence of tokens. Text naturally comes as a sequence of words. An image is a 2D grid of pixels — not a sequence.

**Solution:** Chop the image into patches and treat each patch as a "token."

```text
Text:   "The  cat  sat  on"    → 4 tokens  → embed each → sequence of 4 vectors
Image:  [224 × 224 photo]      → 196 patches → embed each → sequence of 196 vectors
                                  (14 × 14 grid of 16 × 16 pixel patches)

From here on, it's the SAME transformer blocks as text encoders:
attention → residual → LayerNorm → FFN → residual → LayerNorm

The transformer doesn't know (or care) that the "tokens" are image patches.
```

---

### ViT Architecture: Stage by Stage

```text
Image (3 × 224 × 224)
    ↓
Stage 1: Patch + Embed         → 196 vectors (one per patch)
    ↓
Stage 2: Prepend [CLS] token   → 197 vectors
    ↓
Stage 3: Add position vectors  → 197 vectors (now position-aware)
    ↓
Stage 4: 12 transformer blocks → 197 context-aware vectors
    ↓
Stage 5: Take [CLS] output     → 1 vector (768-d)
    ↓
Stage 6: Normalize              → final image embedding
```

---

#### Stage 1: Patch Embedding

Chop the 224×224 image into a grid of 16×16 pixel patches:

```text
224 ÷ 16 = 14 patches per row
14 × 14 = 196 patches total
Each patch: 16 × 16 × 3 channels = 768 raw numbers
```

Each flattened patch is projected to the model's hidden dimension via a **learned linear layer**:

```text
Toy example (4×4 grayscale image, 2×2 patches, project to 3-d):

Original image (4×4):                   4 patches (2×2 each, flattened):
┌─────┬─────┐
│ 1 2 │ 0 1 │                          Patch 0: [1, 2, 0, 1]  (top-left)
│ 0 1 │ 3 2 │                          Patch 1: [0, 1, 3, 2]  (top-right)
├─────┼─────┤                          Patch 2: [2, 1, 1, 2]  (bottom-left)
│ 2 1 │ 0 0 │                          Patch 3: [0, 0, 1, 3]  (bottom-right)
│ 1 2 │ 1 3 │
└─────┴─────┘

Projection matrix W_patch (4 × 3) — learned during training:
[[0.1, 0.2, 0.0],
 [0.3, 0.1, 0.4],
 [0.0, 0.5, 0.1],
 [0.2, 0.0, 0.3]]

Patch 0 embedded = [1, 2, 0, 1] × W_patch
                 = [1×0.1+2×0.3+0×0.0+1×0.2,  1×0.2+2×0.1+0×0.5+1×0.0,  1×0.0+2×0.4+0×0.1+1×0.3]
                 = [0.90, 0.40, 1.10]

Patch 1 embedded = [0, 1, 3, 2] × W_patch
                 = [0+0.3+0+0.4,  0+0.1+1.5+0,  0+0.4+0.3+0.6]
                 = [0.70, 1.60, 1.30]
```

This is analogous to the word embedding lookup in text encoders, but instead of looking up a row in a table (finite vocabulary), we multiply by a matrix (infinite possible patches).

---

#### Stage 2: Prepend [CLS] Token

Just like BERT, prepend a special learnable vector that will accumulate information from all patches via attention:

```text
Before:  [patch_0,  patch_1,  patch_2,  patch_3]         → 4 vectors
After:   [[CLS],  patch_0,  patch_1,  patch_2,  patch_3]  → 5 vectors

[CLS] is a random vector (e.g., [0.05, 0.02, 0.08]), updated during training.
Its job: attend to all patches across all 12 blocks → summarise the whole image.
```

---

#### Stage 3: Add Position Embeddings

Without position info, the transformer can't tell where patches came from. "Top-left corner" vs "bottom-right corner" would look identical to the model.

```text
Position embeddings — learned (same as BERT), one per position:
    pos_0 = [0.01, 0.00, 0.01]   ← for [CLS]
    pos_1 = [0.00, 0.01, 0.00]   ← for patch 0 (top-left)
    pos_2 = [0.01, 0.01, 0.00]   ← for patch 1 (top-right)
    pos_3 = [0.00, 0.00, 0.01]   ← for patch 2 (bottom-left)
    pos_4 = [0.01, 0.00, 0.00]   ← for patch 3 (bottom-right)

Add element-wise:
    [CLS]   + pos_0 = [0.05+0.01, 0.02+0.00, 0.08+0.01] = [0.06, 0.02, 0.09]
    patch_0 + pos_1 = [0.90+0.00, 0.40+0.01, 1.10+0.00] = [0.90, 0.41, 1.10]
    ...etc.
```

Note: positions are 1D (0, 1, 2, ...), not 2D (row, col). The model learns to figure out the 2D layout from these 1D positions during training. This works surprisingly well.

---

#### Stage 4: Transformer Blocks

**Identical to the text encoder (file 06).** Same self-attention, same residual connections, same LayerNorm, same FFN.

```text
One block (same as text encoder):

Input: 5 vectors (CLS + 4 patches), each 3-d (768-d in real ViT)

    Multi-Head Self-Attention:
        Every patch attends to every other patch (global, not local like CNN)
        [CLS] attends to all patches → gathers global information
        Patch 0 (top-left) can directly "see" patch 3 (bottom-right)
            → This is what CNNs need 20+ layers for

    Residual + LayerNorm

    Feed-Forward Network:
        Expand 768 → 3072 → shrink back to 768
        Applied to each vector independently

    Residual + LayerNorm

Output: 5 vectors, same shape, but now context-aware
```

ViT-B/16 stacks **12 of these blocks**, each with its own separate weights — exactly like BERT.

What the blocks learn at different depths:

```text
Early blocks (1-4):    Nearby patches notice each other
                       "This edge connects to that edge"

Middle blocks (5-8):   Patches group into object parts
                       "These patches together form a face"

Late blocks (9-12):    Global understanding
                       "This is a photo of a cat sitting on a mat"
```

---

#### Stage 5: Extract Image Vector

Two options (same choice as text encoders):

```text
1. [CLS] pooling (original ViT):
    v_image = output[0]    ← the CLS vector, which attended to all patches
    Used by: original ViT, CLIP

2. Mean pooling (DINOv2 and others):
    v_image = mean(output[1:])    ← average all patch vectors, excluding CLS
    Often works better in practice for retrieval tasks
```

#### Stage 6: Normalize

Scale to unit length for cosine similarity — same as text encoders.

---

### CNN vs ViT: Side-by-Side

```text
                        CNN (ResNet-50)              ViT-B/16
──────────────────────────────────────────────────────────────────
How it "reads"          Slides filters locally       Every patch sees every patch
                        (3×3 window)                 (global attention)

Building blocks         Conv → ReLU → Pool           Attention → FFN
                                                     (same as text encoder)

Position awareness      Built into the sliding       Must be added explicitly
                        (filter position = image     (learned position embeddings)
                        position)

Long-range relations    Only after many layers        Immediate (attention is global)
                        (slow receptive field growth)

Inductive bias          Strong: locality,            Weak: assumes nothing,
                        translation invariance       must learn everything from data

Data hunger             Works with ~1M images         Needs 14M+ images (or
                        (ImageNet-1K)                 self-supervised pre-training)

Parameters              25M (ResNet-50)               86M (ViT-B/16)
Output dims             2048                          768
```

**Key trade-off:** CNNs bake in useful assumptions about images (nearby pixels matter more, patterns can appear anywhere). ViT assumes nothing and learns everything — more flexible but needs much more data or smarter training (self-supervised).

---

## Part 3: How Image Encoders Are Trained

The architecture (CNN or ViT) is the machine. The training game decides what it learns. There are four main games:

---

### Game 1: Supervised Classification

The original training method. Requires labelled data.

```text
Dataset: ImageNet — 1.3M images, 1000 classes (cat, dog, car, ...)

How it works:
    1. Feed image through encoder → vector (2048-d or 768-d)
    2. Classification head: vector → 1000 probabilities (softmax)
    3. Loss = cross-entropy (predicted class vs true class)
    4. Backprop updates all weights

    Image of a cat
        ↓
    Encoder → [0.08, 0.91, -0.23, ...]
        ↓
    Linear head → P(cat) = 0.85, P(dog) = 0.05, ...
        ↓
    Target: P(cat) = 1.0
        ↓
    Loss = cross-entropy → backprop

For embeddings: after training, throw away the classification head.
The vector before the head is your image embedding.
```

This is how ResNet and the original ViT were trained.

**Limitation:** needs millions of labelled images. ViT especially struggles — with only ImageNet-1K (1.3M images), it underperforms ResNet. Google's original ViT paper used ImageNet-21K (14M images) and JFT-300M (300M images) to make it work.

---

### Game 2: Supervised Fine-Tuning for Similarity (Triplet / Contrastive)

When you have your own labelled pairs (e.g., "same product" vs "different product"):

```text
Training triplets:
    Anchor:   photo of blue Nike shoe
    Positive: different angle of same blue Nike shoe    ← should be close
    Negative: photo of red Adidas shoe                 ← should be far

Triplet loss:
    L = max(0, sim(anchor, negative) − sim(anchor, positive) + margin)

    margin ≈ 0.2 — enforces a minimum gap between positive and negative

What this does:
    Adjusts the encoder's weights so that:
    sim(anchor, positive) > sim(anchor, negative) + 0.2
```

Used when you need domain-specific similarity (product photos, face recognition, etc.) and have labelled pairs.

---

### Game 3: Self-Supervised Contrastive (SimCLR, MoCo, DINO)

No labels needed. Create training signal from the images themselves.

**Core idea:** Two different augmentations of the SAME image should produce SIMILAR vectors. Augmentations of DIFFERENT images should produce DIFFERENT vectors.

```text
Step 1: Take one image
Step 2: Create two random augmentations:
    Aug 1: random crop + colour jitter + horizontal flip
    Aug 2: different random crop + blur + grayscale

    Original          Aug 1                 Aug 2
    ┌──────────┐     ┌──────┐             ┌──────────┐
    │  photo   │  →  │cropped│       →    │ blurred  │
    │  of cat  │     │colour │            │ greyscale│
    └──────────┘     │jitter │            │  crop    │
                     └──────┘             └──────────┘

Step 3: Feed both through encoder → vector_1, vector_2
Step 4: Loss pushes vector_1 close to vector_2,
        pushes vector_1 away from vectors of other images in the batch
```

#### SimCLR (2020)

```text
Straightforward contrastive:
    - Both augmentations go through the SAME encoder
    - Positive pair: two augmentations of the same image
    - Negative pairs: all other images in the batch
    - Loss: InfoNCE (similar to softmax — make the positive pair
      score higher than all negative pairs)

    L = −log[ exp(sim(v₁, v₂) / τ) / Σⱼ exp(sim(v₁, vⱼ) / τ) ]
        where τ = temperature (controls sharpness)

    Problem: needs very large batches (4096+) for enough negatives.
    Smaller batches → fewer negatives → worse training signal.
```

#### MoCo (2020)

```text
Solves SimCLR's batch size problem:
    - Maintains a QUEUE of recent vectors as negatives
    - Queue size: 65,536 (much more negatives than any batch)
    - Uses a momentum-updated encoder for the queue entries:
        encoder_momentum = 0.999 × encoder_momentum + 0.001 × encoder
    - This keeps queue entries consistent even though the encoder is changing

    Result: works with normal batch sizes (256), still gets lots of negatives.
```

#### BYOL (2020) — No Negatives At All

```text
Uses a teacher-student setup:
    Student: the model being trained
    Teacher: a slowly-updated copy of the student (exponential moving average)

    Aug 1 → Student encoder → student prediction
    Aug 2 → Teacher encoder → teacher target

    Loss: student must match teacher (no negatives needed)
    Only the student gets gradient updates.
    Teacher updates slowly: teacher = 0.996 × teacher + 0.004 × student

Why doesn't the model collapse (map everything to the same vector)?
    - Student has an extra "predictor" head that teacher doesn't
    - The asymmetry (predictor + stop-gradient on teacher) prevents collapse
    - Avoids the need for large batches entirely
```

#### DINO (2021) — Self-Supervised ViT

```text
Similar teacher-student setup as BYOL, but designed for ViTs.

Key innovation — multi-crop strategy:
    From one image, create:
    - 2 "global" crops (covering 50-100% of the image)
    - Several "local" crops (covering 5-30% of the image)

    All crops go through the student.
    Only global crops go through the teacher.

    Student must predict teacher's output for GLOBAL crops from LOCAL crops.

    This forces the student to understand the big picture from a small piece.
    Seeing just a cat's ear? You should still know it's a cat.

    Teacher updates slowly: teacher = 0.996 × teacher + 0.004 × student
```

---

### Game 4: Masked Image Modelling (MAE)

The image equivalent of BERT's masked language model.

```text
Step 1: Chop image into patches (like ViT)
Step 2: Randomly MASK 75% of patches
Step 3: Encoder processes only the visible 25% of patches
Step 4: A small decoder tries to reconstruct the missing pixels

    Original patches:          Masked (75%):           Reconstruct:
    ┌──┬──┬──┬──┐             ┌──┬──┬──┬──┐           ┌──┬──┬──┬──┐
    │p0│p1│p2│p3│             │p0│██│██│p3│           │p0│p1│p2│p3│
    ├──┼──┼──┼──┤      →      ├──┼──┼──┼──┤     →     ├──┼──┼──┼──┤
    │p4│p5│p6│p7│             │██│██│p6│██│           │p4│p5│p6│p7│
    └──┴──┴──┴──┘             └──┴──┴──┴──┘           └──┴──┴──┴──┘

    Loss = MSE between predicted pixels and actual pixels for masked patches

Why 75% masking (vs BERT's 15%)?
    Images have much more redundancy than text.
    Neighbouring patches look very similar — masking only 15% is too easy,
    the model just interpolates from neighbours without learning anything deep.
    75% forces it to actually understand the content.

For embeddings: after pre-training, throw away the decoder.
    The encoder has learned rich visual features.
    Optionally fine-tune with contrastive loss for better similarity vectors.
```

---

### Labelled vs Unlabelled: Quick Comparison

| Aspect | **Labelled (Supervised)** | **Unlabelled (Self-supervised)** |
| --- | --- | --- |
| **How the positive is chosen** | Another image with a human-defined identity: same SKU, same person, same scene | A different augmentation of the **same image** (crop, blur, colour-jitter) |
| **How the negative is chosen** | An image known to be different (different SKU/person) | Any other image in the batch; no labels needed |
| **What encoder learns** | "All photos of these shoes → close vectors" (instance recognition) | "This crop and that crop of the same photo → close vectors" (augmentation invariance) |
| **When to prefer** | You have labelled pairs and need domain-specific similarity | You have thousands of images but no labels |

---

## Part 4: Popular Models and How to Choose

### DINOv2 (2023) — Current Best Open-Source Image Encoder

DINOv2 deserves its own section because it's the current state of the art for general visual features.

```text
What it combines:
    1. ViT architecture (ViT-S/B/L/g — small to giant)
    2. DINO's teacher-student game
    3. iBOT's masked image modelling — like MAE, but predicts teacher's
       token representations instead of raw pixels (harder, more useful)
    4. Trained on LVD-142M — a curated dataset of 142M diverse images
       (not ImageNet — custom, deduplicated, diverse)

Why it's special:
    - Completely self-supervised (zero human labels)
    - Features are so good they work out-of-the-box for:
        → Image classification (just attach a simple linear head)
        → Depth estimation
        → Segmentation
        → Retrieval / similarity search
    - The patch-level features are spatially meaningful:
        each patch output vector "knows" what object/part it covers

Sizes:
    ViT-S/14:  21M params,  384-d vector
    ViT-B/14:  86M params,  768-d vector
    ViT-L/14: 300M params, 1024-d vector
    ViT-g/14:   1B params, 1536-d vector
```

---

### Model Comparison Table

```text
| Model           | Arch.       | Dims | Training             | Params | Best for                          |
| --------------- | ----------- | ---- | -------------------- | ------ | --------------------------------- |
| ResNet-50       | CNN         | 2048 | Supervised (ImageNet)| 25M    | Lightweight, edge, legacy         |
| EfficientNet-B7 | CNN         | 2560 | Supervised (ImageNet)| 66M    | Best CNN accuracy/efficiency      |
| ViT-B/16        | Transformer | 768  | Supervised (ImageNet)| 86M    | General purpose with fine-tuning  |
| DINOv2 ViT-B/14 | Transformer | 768  | Self-supervised      | 86M    | Best general features (no labels) |
| DINOv2 ViT-L/14 | Transformer | 1024 | Self-supervised      | 300M   | Best quality (needs GPU)          |
| CLIP ViT-L/14   | Transformer | 768  | Text-image contrastive| 300M  | Multimodal search (next file)     |
| MAE ViT-L/16    | Transformer | 1024 | Masked autoencoder   | 300M   | Fine-tuning on domain data        |
```

---

### How to Choose

```text
"I'm prototyping / learning"
    → DINOv2 ViT-B/14 (best features out of the box, one line of code)
    → or ResNet-50 if you want something simpler and lighter

"I need best quality"
    → DINOv2 ViT-L/14 (open-source, self-supervised)
    → Fine-tune on your domain data if needed

"I need to search images with text queries"
    → CLIP (covered in the next file — multimodal encoders)

"I have domain-specific images (medical, satellite, etc.)"
    → Start with DINOv2, fine-tune with contrastive pairs from your domain

"I need to run on mobile / edge"
    → EfficientNet-B0 or MobileNetV3 (small CNNs designed for efficiency)
    → or DINOv2 ViT-S/14 (smallest ViT variant, 21M params)

"I have lots of unlabelled images, no labels"
    → DINOv2 or MAE pre-training on your images
    → or SimCLR/BYOL if you prefer a contrastive approach

"I have labelled pairs (same product / different product)"
    → Fine-tune ResNet-50 or ViT with triplet loss on your pairs
```

---

### The Connection: Text Encoder ↔ Image Encoder

```text
Text Encoder (file 06):                 Image Encoder (ViT):
─────────────────────────               ─────────────────────────
Tokenizer → token IDs                  Patch + flatten → patch vectors
Embedding lookup table                  Linear projection (W_patch)
Add position embeddings                 Add position embeddings
12 transformer blocks                   12 transformer blocks (identical!)
    (attention → residual → LN → FFN)      (attention → residual → LN → FFN)
Pooling (mean or [CLS])                 Pooling (mean or [CLS])
Normalize                               Normalize
    ↓                                       ↓
768-d text vector                        768-d image vector

The ONLY difference is Stage 1: how you turn raw input into a sequence of vectors.
Everything from Stage 2 onward is the same architecture.

This is why CLIP works: if text encoder and image encoder both use the same
kind of transformer and output vectors of the same dimension, you can train
them to put similar text and images close together in the SAME vector space.
```

---

### Evolution Timeline

```text
2012  AlexNet         — first deep CNN to win ImageNet, proved deep learning works
2015  ResNet          — skip connections enabled 50-150+ layer CNNs
2017  EfficientNet    — best accuracy/efficiency trade-off for CNNs
2020  ViT             — applied transformers to images, needed huge labelled datasets
2020  SimCLR / MoCo   — self-supervised training for CNNs (no labels)
2020  BYOL            — self-supervised without even needing negatives
2021  DINO            — self-supervised ViT with teacher-student
2022  MAE             — masked image modelling (BERT for images)
2023  DINOv2          — combined DINO + iBOT + curated data = current best features
2024  SigLIP / EVA-02 — pushing further with better training recipes

The trend: self-supervised ViTs are replacing supervised CNNs.
No labels needed, better features, same transformer blocks as text.
```

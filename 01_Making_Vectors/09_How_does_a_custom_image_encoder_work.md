## How do image encoders work?

| Stage                                                                          | What goes in & out                                                                            | Why it exists                                                 | Typical size notes                                           |
| ------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------- | ------------------------------------------------------------- | ------------------------------------------------------------ |
| **1. Resize & scale**                                                          | Photo → 224 × 224 pixels, 3 colour channels.                                                  | Makes every image the same size and value range.              | **3 × 224 × 224** grid.                                      |
| **2. First pattern layer**                                                     | Slides 64 tiny 7 × 7 filters over the picture.                                                | Detects simple blobs and edges.                               | **64 channels × 112 × 112** (size halves because of stride). |
| **3. Four “pattern-refine” stages**<br>(each stage = a small stack of filters) | Re-scan & combine earlier patterns to form corners → textures → object parts → whole objects. | Builds up complexity step by step.                            | End of stage 4: **2048 channels × 7 × 7** little grids.      |
| **4. *Global-average-pool* (the sponge squeeze)**                              | Averages each 7 × 7 grid down to one number.                                                  | Turns the 3-D pile into a flat list; order no longer matters. | **1 × 2048** vector.                                         |
| **5. Normalise (optional)**                                                    | Scales the vector to unit length.                                                             | Makes cosine similarity consistent.                           | Still **1 × 2048**.                                          |

- Output: one 2 048-number vector for the picture.


Below is a “pocket-size” CNN that takes a 4 × 4 grayscale picture and turns it into a single vector.
```text
I = 4×4 matrix
[[1, 2, 0, 1],
 [0, 1, 3, 2],
 [2, 1, 0, 0],
 [1, 2, 1, 3]]
```
(One channel → one matrix.)

**Convolution layer (channels increase)**

Important information for understanding convolution
| Word                              | One-line picture                                                                                                                                         | Think of it as…                                                                    |
| --------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| **Channel**                       | A *stacked page* that holds one kind of information for **every pixel**.  In a colour photo the red, green, and blue pages are the first three channels. | A separate “layer of ink” in the same image.                                       |
| **Filter** (also called a kernel) | A tiny patch of numbers—say **3 × 3 × C**—that is slid over the image.  It multiplies its own numbers with what it sees underneath and sums the result.  | A cookie-cutter that looks for a specific pattern (horizontal edge, dot, texture). |


Now consider, 
```text
Filter A (2×2)               Filter B (2×2)
[[ 1,  0],                   [[ 0, -1],
 [ 0,  1]]                    [ 1,  0]]
```
Each filter will slide over every 2 × 2 patch of I (stride = 1, no padding).

Convolution is given as:
```math
out[c, y, x] = Σ_{dy,dx,k}   in[k, y+dy, x+dx] * W[c,k,dy,dx]   + b[c]  
```
but it translates to moving the filter kernel over the entire matrix and taking dot product to get the new value.

Note:
Depth of a filter always matches the number of input channels.
Input tensor: 3 × 224 × 224 (one “page” each for Red, Green, Blue).
Conv 1 filter: 7 × 7 × 3 —- seven-by-seven patch, three layers deep.
The filter slides across the picture.
At every (y, x) location it multiplies its own 7×7×3 numbers with the 7×7×3 block of pixels underneath, sums them, adds a bias → outputs one single number at (y, x) in its personal result page.
Hence for c filters each of size (f * f * k), and an image (k * h * w), the output is (c * h_out * w_out) where h_out = floor((h - f + 2*padding) / stride) + 1

Here, the output tensor after this layer: 2 channels × 3 × 3
```text
A = [[2, 4, 1],
     [1, 4, 4],
     [3, 2, 1]]

B = [[ 0, -1, -1],
     [ 3,  0,  2],
     [ 1,  0, -2]]
```
Key point: channels jumped from 1 to 2 because we introduced 2 different filters.
* What changes? The channel count (c) can grow, giving more pattern types.
* What stays? Height and width, unless we also slide by 2 pixels at a time (stride = 2).

After convolution, next is an **activation function**
```text
Activation (ReLU)
ReLU(x) = max(0, x)

A_relu = [[2, 4, 1],
          [1, 4, 4],
          [3, 2, 1]]

B_relu = [[0, 0, 0],
          [3, 0, 2],
          [1, 0, 0]]
```
Note: Negatives in B became zeros. This helps us focus on the important features and avoid small shrinking/negative values.

The next step is **Spatial down-sampling**
The simplest form is **average pooling**: slide a 2 × 2 window with stride 2 and take the mean of each window (shrinks spatial dims by half).
That halves height & width.
Another common form is **Max-pool**: look at a 2 × 2 window, keep the biggest number.

The last step is the **Global-Average-Pooling (GAP)**
This simply means to take the mean of each channel’s remaining grid:
```text
If you had 2 channels after convolution and pooling,
A_small = [[3, 2],
           [2, 3]]

B_small = [[0, 1],
           [1, 0]]

GAP is:
v_A = (3 + 2 + 2 + 3) / 4 = 2.5
v_B = (0 + 1 + 1 + 0) / 4 = 0.5

Vector = [2.5, 0.5]
```
The encoders convolution and pooling layers result in ~ (2048*7*7)
The encoders GAP output is then a 2048 size vector.

### Summary:
In images: each convolution looks at a slightly bigger area—edges → corners → textures → object parts—while stride/pooling zooms out. The last 7 × 7 × 2048 map says “where—and how strongly—did I see each high-level part?”. Averaging (GAP) collapses where and keeps how strong, yielding the fixed-length vector.


| Stage               | Plain-English job                                                                                                        |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| **Convolution**     | Look for little patterns (edges, blobs) and create a new *channel* for each pattern type.  More filters → more channels. |
| **ReLU**            | Keep strong pattern matches, erase weak/negative ones.                                                                   |
| **Stride 2 / Pool** | Zoom out: same field of view now covered by fewer pixels, so next filters can see *bigger* shapes.                       |
| **GAP**             | Throw away *where* patterns occurred, keep *how much* each pattern appeared → gives a fixed-size summary.                |


General Neural Net Structure:

| Stage                     | Operation          | How many filters     | Stride          | Output size *(channels × height × width)* |
| ------------------------- | ------------------ | -------------------- | --------------- | ----------------------------------------- |
| **Input**                 | —                  | —                    | —               | **3 × 224 × 224**                         |
| **Conv 1**                | 7 × 7 filter       | 64                   | 2 (+ pad 3)    | 64 × 112 × 112                            |
| **Max-pool**              | 3 × 3              | —                    | 2 (+ pad 1)    | 64 × 56 × 56                              |
| **Conv 2 block**          | 3 bottleneck layers| 64 → 64 → **256**    | 1               | 256 × 56 × 56                             |
| **Conv 3 block**          | 4 bottleneck layers| 128 → 128 → **512**  | 2 (first layer) | 512 × 28 × 28                             |
| **Conv 4 block**          | 6 bottleneck layers| 256 → 256 → **1024** | 2               | 1024 × 14 × 14                            |
| **Conv 5 block**          | 3 bottleneck layers| 512 → 512 → **2048** | 2               | 2048 × 7 × 7                              |
| **GAP** (Global-Avg-Pool) | average over 7 × 7 | —                    | —               | **2048 × 1 × 1**                          |
| **Flatten / L2-norm**     | —                  | —                    | —               | **2048-D vector**                         |

Reading the table in plain language
- Conv 1 (64 × 112 × 112) – 64 edge detectors scan the RGB image; output is 64 grey pages, each 112 × 112 pixels.
- Max-pool (64 × 56 × 56) – we zoom out; each page loses half its width and height.
- Conv 2 block – tiny filters combine edges into corners/texture patches; 256 pattern pages now.
- Conv 3/4/5 blocks – keep repeating: zoom out a bit, then add more pattern pages. By the last block each “pixel” covers a big chunk of the original photo, and we have 2048 different pattern types.
- Global-average-pool squeezes every 7 × 7 page into one number, leaving a tidy 2048-number list that summarises how much each pattern occurred anywhere in the image.
- That list is the embedding vector you store and compare with cosine distance.

### Residual (Skip) Connections — What Makes ResNet “Res”Net

The “Res” in ResNet stands for **Residual**. This is the defining innovation.

**The problem:** Very deep CNNs (20+ layers) actually perform *worse* than shallower ones — not because of overfitting, but because gradients vanish during training, making deeper layers unable to learn.

**The fix:** Instead of learning the full output `H(x)`, each block learns the *residual* `F(x) = H(x) - x`, then adds the input back:

```text
output = F(x) + x       ← the skip connection
```

Visually:
```text
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

**Why it works:** If a layer has nothing useful to learn, it can simply learn F(x) = 0, making the block a pass-through. This means adding more layers can never hurt — at worst, they learn to do nothing. Gradients also flow directly through the skip connection, solving the vanishing gradient problem.

**Dimension mismatch:** When a block changes the number of channels (e.g., 256 → 512), the skip connection uses a 1×1 convolution to match dimensions before adding.

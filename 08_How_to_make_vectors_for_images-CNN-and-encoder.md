## How we turn images into a single vector?

The story parallels the procss for text: 
- pick or train an encoder, 
- feed every picture through it once, store the resulting vector in a vector DB, 
- and compare with cosine distance at query time.


### 1 Off-the-shelf CNN pooling (the “simple pooling” analogue)

- Encoder – a convolutional network pre-trained on ImageNet, e.g. ResNet-50.
- Global Average Pooling – replace the final classification head with a spatial average over the last feature map, giving a fixed-length vector (2048 d for ResNet-50).

#### What global average pooling really does?

A convolutional network (e.g., ResNet-50) finishes its last block with a feature map shaped:
```text
[C, H, W]   # channels, height, width  
# ResNet-50 → [2048, 7, 7]  
```

Each channel is a 7 × 7 activation grid that tells “how strongly the image shows feature c at position (h, w).”
Global Average Pooling (GAP) collapses the spatial dimensions by taking the mean over every (h, w) cell:
```math
v_c = (1 / (H * W)) * Σ_{h=1..H} Σ_{w=1..W} F[c, h, w]
```
So the 2048 × 7 × 7 tensor shrinks to a 2048-length vector.
Originally ResNet feeds that vector into a 2048 → 1000 fully-connected layer for ImageNet classes.
For embeddings we drop the FC layer and keep the 2048-d GAP output as the image’s fixed-length representation.

- Pros: one line of TorchVision, no extra data.
- Cons: tuned for ImageNet classes; may blur fine product details (colour shades, logo variants).


### Fine-tuned CNN for visual similarity

Exactly the same network, but you re-train the last few layers (or the whole backbone) on triplets of your own product photos:
```text
(anchor A, positive P, negative N)
```
Loss (Triplet) pushes A-P close, A-N far:
```math
L = max(0, cos(A, N) – cos(A, P) + α);  where α ≈ 0.2  
```

- Preferred when you can label pairs as “same SKU vs. different SKU”.


### Self-supervised contrastive encoders (SimCLR, MoCo, BYOL)

- No labels at all. 
- Create two random augmentations of the same image, by cropping, rotating, zooming in/out, etc.
- treat them as a positive pair, and use the rest of the batch as negatives. 

Works with the InfoNCE loss
```math
L = – log [exp(cos(v₁, v₂)/τ) / Σ_j exp(cos(v₁, v_j)/τ)]  
```

- learns a backbone that groups visually similar patterns (texture, shape) even on unlabelled catalog shots.


### Labelled vs Unlabelled learning

| Aspect                          | **Labelled (Supervised) triplet**                                                                        | **Unlabelled (Self-supervised) triplet**                                                                                  |
| ------------------------------- | -------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| **How the positive is chosen**  | Another image that shares a *human-defined identity* with the anchor: same SKU, same person, same scene. | A *different augmentation* of the **same physical image** (crop, colour-jitter, blur).                                    |
| **How the negative is chosen**  | An image known to be *different* from the anchor (different SKU/person).                                 | Any *other* image in the batch; no labels needed.                                                                         |
| **What the encoder must learn** | View-invariant *instance recognition*: “all photos of these shoes map close together.”                   | Augmentation-invariant *feature extraction*: “this colour-shifted crop and the original still represent the same object.” |
| **Typical loss**                | Triplet loss or NT-Xent with labelled pairs.                                                             | NT-Xent / InfoNCE where positives are augmentations.                                                                      |
| **When to prefer**              | You have ≥2 photos per product/person and reliable labels.                                               | You have thousands of images but *no* labels; cheaper to collect.                                                         |


### Recommended path for product photos
| Data you have                                    | Encoder to start with                    | Steps                                                      |
| ------------------------------------------------ | ---------------------------------------- | ---------------------------------------------------------- |
| < 1 k images, no labels                          | *Pre-trained ResNet-50*                  | Just global-average-pool them.                             |
| Few k – few 10 k images, labels (“same product”) | *ResNet-50 fine-tuned with triplet loss* | Freeze lower layers, train 5–10 epochs.                    |
| 50 k + unlabelled shots                          | *SimCLR or BYOL* self-supervised         | Train backbone for 100–200 epochs on random crops.         |
| Images **and** textual metadata                  | *CLIP / OpenCLIP*                        | Possibly fine-tune CLIP on your pairs for sharper matches. |

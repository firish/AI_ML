# Text–Image Matching Models (e.g. CLIP / ALIGN)

The goal: **Embed a photo and its *true* caption at nearly the same coordinates in vector space, while pushing mismatched pairs far apart.**

---

## 1 Two Encoders Running Side-by-Side

| **Image Factory** | **Text Factory** |
|-------------------|------------------|
| *Image encoder* – a CNN (or Vision Transformer).<br>**Output:** single vector (e.g., 512 / 768 dims). | *Text encoder* – a Transformer LM.<br>**Output:** single vector of the **same** length. |

Nothing is shared between them except the training signal.

---

## 2 Training Data = Huge Piles of Pairs

```
(image of a golden retriever, "A happy dog playing in the grass")
(image of the Eiffel Tower,   "The Eiffel Tower at sunset")
…
```

Captions come “for free” from the web (alt-text, social media, etc.).

---

## 3 The Contrastive Learning Game (per mini-batch)

1. **Batch of B pairs** (toy example below uses `B = 3`):

   ```
   (I₀, T₀)   # dog
   (I₁, T₁)   # tower
   (I₂, T₂)   # pizza
   ```

2. **Encode**

   ```
   v_I0, v_I1, v_I2 ← image encoder
   v_T0, v_T1, v_T2 ← text encoder
   ```

3. **Normalize** vectors to unit length (cosine-friendly).

4. **Similarity matrix**  
   `S = v_I · v_Tᵀ`

   |     | **T₀** | **T₁** | **T₂** |
   |-----|-------:|-------:|-------:|
   | **I₀** | 0.92 | 0.10 | 0.05 |
   | **I₁** | 0.12 | 0.88 | 0.07 |
   | **I₂** | 0.05 | 0.09 | 0.95 |

5. **Contrastive loss**

   ```
   Loss_row = CE(softmax(row_i), target = i)
   Loss_col = CE(softmax(col_j), target = j)
   Total    = (Loss_row + Loss_col) / 2
   ```

6. **Back-propagate** through *both* encoders – correct pairs climb, wrong pairs sink.

After millions of batches the encoders share a **joint semantic space**.

---

## 4 Using a Trained Model

### A. Image ↔ Caption Search

~~~python
query = "red running shoes"
v_q   = text_encoder(query)
nearest_imgs = ann_search(v_q, image_vector_db)
~~~

### B. Zero-Shot Classification

1. Prepare text prompts: `"cat"`, `"dog"`, `"car"`, …  
2. Encode the image once and every prompt.  
3. Pick the prompt with highest cosine → predicted class.

### C. Multi-Modal Chat / RAG

Because text **and** images share one space, you can:

* Retrieve images that match a text query.  
* Retrieve captions that match an uploaded image.  
* Feed both to a downstream LLM.

---

## 5 Why It Generalizes

The model learns *relative geometry*, not raw pixels or n-grams.  
`“Eiffel Tower” ↔ tall-thin-metal landmark` in both modalities, so new views or phrasings still align.

---

## Tiny Code Sketch (PyTorch-like)

~~~python
img_encoder = ClipImageEncoder()          # pre-trained
txt_encoder = ClipTextEncoder()

img_vec = img_encoder("tower.jpg")        # shape (512,)
txt_vec = txt_encoder("The Eiffel Tower")

similarity = (img_vec @ txt_vec).item()   # ≈ cosine
print("similarity =", similarity)
~~~

High score (~0.9) → good match; mismatched caption scores low.

---

## Bottom Line

* Two separate encoders.  
* One simple “match-your-partner” contrastive game.  
* Result: a **shared vector language** where semantically similar images and texts land next to each other—enabling search, zero-shot tasks, and multi-modal reasoning.

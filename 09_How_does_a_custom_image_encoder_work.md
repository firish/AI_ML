## How do image encoders work?

| Stage                                                                          | What goes in & out                                                                            | Why it exists                                                 | Typical size notes                                           |
| ------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------- | ------------------------------------------------------------- | ------------------------------------------------------------ |
| **1. Resize & scale**                                                          | Photo → 224 × 224 pixels, 3 colour channels.                                                  | Makes every image the same size and value range.              | **3 × 224 × 224** grid.                                      |
| **2. First pattern layer**                                                     | Slides 64 tiny 7 × 7 filters over the picture.                                                | Detects simple blobs and edges.                               | **64 channels × 112 × 112** (size halves because of stride). |
| **3. Four “pattern-refine” stages**<br>(each stage = a small stack of filters) | Re-scan & combine earlier patterns to form corners → textures → object parts → whole objects. | Builds up complexity step by step.                            | End of stage 4: **2048 channels × 7 × 7** little grids.      |
| **4. *Global-average-pool* (the sponge squeeze)**                              | Averages each 7 × 7 grid down to one number.                                                  | Turns the 3-D pile into a flat list; order no longer matters. | **1 × 2048** vector.                                         |
| **5. Normalise (optional)**                                                    | Scales the vector to unit length.                                                             | Makes cosine similarity consistent.                           | Still **1 × 2048**.                                          |

- Output: one 2 048-number vector for the picture.


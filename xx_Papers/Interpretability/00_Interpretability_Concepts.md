## Interpretability — Core Concepts & Language

A reference for the mental models, vocabulary, and techniques used in mechanistic
interpretability research. Read this before diving into individual papers.

---

## 1. MLP = FFN (Same Thing, Different Name)

```text
"MLP" (multi-layer perceptron) and "FFN" (feed-forward network) are the same block
in a transformer. Different naming traditions, identical machinery.

What it is:
    input (d_model) → up_proj (d_model → 4×d_model) → activation (GELU/SwiGLU) → down_proj (4×d_model → d_model) → output

Every transformer layer = one attention block + one MLP block.

When interpretability papers say "MLP block," they mean exactly this.
Nothing new.
```

---

## 2. The Residual Stream

```text
You know transformers as stacks of attention + MLP blocks. Here's the reframe
that interpretability people use — mathematically identical, but much more
useful for thinking about representations.

For each token, the model maintains a single vector (size d_model) that gets
passed up through all the layers. This vector is the "residual stream."

What happens at each layer:
    1. Attention block reads the stream → computes something → ADDS result back
    2. MLP block reads the stream → computes something → ADDS result back
    3. Next layer does the same

In code (you already know this):
    stream ← stream + Attention(LayerNorm(stream))
    stream ← stream + MLP(LayerNorm(stream))

That += is the key insight. Nothing overwrites — everything ADDS.
```

### Why "writes by addition" matters

```text
Because addition is linear, you can decompose the residual stream at any
layer as a literal sum of every contribution that came before:

    stream_at_layer_L = token_embedding
                      + attention_0_output
                      + mlp_0_output
                      + attention_1_output
                      + mlp_1_output
                      + ...
                      + attention_{L-1}_output
                      + mlp_{L-1}_output

Every block's contribution is still "in there" as an additive term.
Nothing has been destroyed, only added to.

This is why the stream works as a communication channel between blocks:
    - Earlier blocks WRITE information into it (by adding their output)
    - Later blocks READ it out (by taking it as their input)
    - You can attribute what's in the stream to specific blocks
```

### Activations vs. Weights

```text
For a 1000-token prompt through a 64-layer model:
    → 1000 × 64 vectors of size d_model
    → Each one is a snapshot of "what the model represents about token t after layer L"
    → These are called ACTIVATIONS

Activations = the content the model is currently "thinking about"
Weights     = the fixed machinery that reads from and writes to the stream

Representations live in activations, not weights.
Weights are the same for every input. Activations change per input.
```

---

## 3. Activation Space

```text
At any given (layer, token position), you have one d_model-dimensional vector.

That vector lives in a d_model-dimensional vector space.
This is "activation space" (at that layer).

For a model with d_model = 4096:
    → Each activation is a point in 4096-dimensional space
    → Different inputs land at different points
    → The geometry of this space encodes what the model "knows"
```

---

## 4. The Linear Representation Hypothesis

```text
The key empirical finding behind most of modern interpretability:

    High-level concepts the model uses are encoded as specific DIRECTIONS
    in activation space.

Concretely:
    There exists some unit vector v (length d_model) — call it the "afraid direction" —
    such that:
        dot_product(v, activation) is LARGE  → model is processing fear-related content
        dot_product(v, activation) is SMALL  → model is not

Multiple concepts coexist in the same vector by occupying different directions,
the way different frequencies coexist in a single audio signal.

What "linear" means:
    - The concept lives along a straight line through the origin
    - You read it off with a dot product (a linear operation)
    - It's NOT stored in any particular neuron
    - It's spread across all d_model coordinates
    - Only the specific direction matters

When a paper says "we extracted a vector for 'desperate'":
    → They found a direction v_desperate in d_model-dim space at a specific layer
    → Projecting any activation onto it tells you how "desperate" the model
      currently represents the context as
```

---

## 5. How They Find a Direction (Contrastive Activation Extraction)

```text
The trick is averaging + contrast.

Step 1: Collect many examples where the concept IS active
    (e.g., 1200 stories about desperation)

Step 2: For each example, run through the model, grab residual stream
    at your chosen layer, average over token positions → one vector per example

Step 3: Average across all examples → centroid of "concept-active" activations

Step 4: Subtract a generic-text baseline
    → Left with what's DISTINCTIVELY that concept, not generic language

Why averaging works:
    Across all desperation stories, lots varies — topic, syntax, vocabulary.
    But the one thing they share is desperation.
    When you average, unrelated stuff washes out (random directions cancel).
    The shared "desperation" component survives. That's your direction.

This is sometimes called "difference-in-means" or "contrastive activation" method.
```

---

## 6. Probes vs. Vectors (Same Object, Two Uses)

```text
Same direction, two names depending on what you do with it:

PROBE (reading):
    Take the direction v. Dot-product it with activations.
    The scalar output tells you "how present is this concept right now?"
    You're MEASURING the model's internal state.

    score = v · activation    ← high = concept is active

VECTOR (writing / steering):
    Take the same direction v. ADD it into the residual stream.
    You're MODIFYING the model's internal state.

    activation_new = activation + α · v

Same object. Probe = read mode. Vector = write mode.
```

---

## 7. Steering

```text
Once you have a direction v for a concept, you can intervene:

    During a forward pass, at chosen positions:
        activation_modified = activation + α · v

You're injecting more of that concept into the running representation.
Downstream layers read this modified stream and behave accordingly.

Key properties:
    - Weights are NOT modified. You're nudging activations, not retraining.
    - Cheap: one vector addition per injection point.
    - Surgical: only that direction changes (in principle).
    - α is the steering strength, reported as fraction of typical activation magnitude.
      α = 0.05 means the injected vector is 5% the size of a normal activation.

Negative α:
    α < 0 SUPPRESSES the concept rather than amplifying it.
    "Steering with α = −0.5 on the calm vector" = subtracting calmness,
    operationally similar to amplifying anti-calm.
```

### Where to inject (steering positions)

```text
No single "correct" set of positions. Different choices answer different questions.

1. All token positions, all the time
    → Bluntest version. Global mood shift for entire forward pass.
    → Downside: heavy-handed, injects even on irrelevant tokens (punctuation,
      boilerplate), can produce artifacts at higher α.

2. Only on generated (assistant) tokens
    → Most common for behavioral questions.
    → Model reads prompt normally, generates response while being nudged.
    → "Will the model agree with my incorrect claim under sycophancy steering?"

3. Only on a specific span
    → Inject only at tokens of a particular substring.
    → Surgical: "did making the model feel emotion E while reading option A
      shift its preference toward A?"

4. Only at one token (e.g., last prompt token)
    → "Set the initial mood, then let it run."
    → Modified state propagates via attention and KV cache.

Layer range:
    Often inject across several adjacent layers (e.g., layers 60-70),
    not just one, because concepts have representational footprint across a band.

Sweep, don't guess:
    Standard practice: sweep α over {0, 0.05, 0.1, 0.2, 0.4, 0.8}.
    Plot behavioral metric against α.
    Look for regime where behavior shifts but model stays coherent.
    Above some threshold (~α=1) model degrades into nonsense.
```

---

## 8. Layer Roles

```text
Layers are NOT all equivalent. They specialize:

Early layers (first ~⅓):
    Surface features — tokenization artifacts, syntax, local patterns.

Middle layers (~⅓ to ~⅔):
    Abstract concepts build up — semantics, relationships, meaning.

Late layers (last ~⅓):
    Convert representations back into next-token-prediction logits.

Why interpretability work targets ~⅔ depth:
    Late enough to be abstract (concepts are fully formed).
    Early enough to causally influence many downstream computations.
    Empirically where high-level concepts are most cleanly represented.
```

---

## 9. The Unembedding Matrix (Logit Lens)

```text
The very last step of the model:
    final_layer_residual_stream × unembedding_matrix = logit per vocab token

If you take ANY direction in residual-stream space and push it through
the unembedding matrix, you get a logit distribution over the vocabulary.

    "Which tokens would this direction promote if it dominated the final layer?"

This is a sanity check:
    Push the "desperate" direction through unembedding →
        ↑ promoted tokens: desperate, urgent, bankrupt, helpless
        ↓ suppressed tokens: pleased, enjoying, delighted

    Confirms the direction actually corresponds to the concept linguistically.

"Logit lens" is the technique of applying this at intermediate layers,
not just the final one — to peek at what the model would predict if
you stopped computation early at layer L.
```

---

## 10. Cosine Similarity Between Directions

```text
Two directions in activation space that point similarly encode similar concepts.

    cos(v_fear, v_anxiety)  ≈ high positive   → related concepts
    cos(v_joy, v_sadness)   ≈ negative         → opposing concepts
    cos(v_fear, v_tylenol)  ≈ near zero        → unrelated concepts

This is the model telling you its internal organization of concepts
resembles the human one. You can build entire semantic maps from
pairwise cosine similarities between extracted directions.
```

---

## 11. Neurons vs. Features (The Core Problem)

```text
Naive hope: each neuron in an MLP represents one concept.
    Neuron 347 = "dogs", neuron 1042 = "French", etc.

Reality: this almost never holds. Most neurons are POLYSEMANTIC —
    a single neuron activates for multiple unrelated concepts.

    Neuron 347 might fire for: dogs, the color brown, AND the word "fetch"
    in a programming context. No single clean meaning.

Why? The model has more concepts to represent than it has neurons.
If d_model = 4096 but the model needs to track 100,000+ concepts,
it can't dedicate one neuron per concept. It has to compress.

This is why interpretability moved from studying individual neurons
to studying DIRECTIONS (linear combinations of neurons). A feature
is a direction in activation space, not a single neuron.
```

---

## 12. Superposition

```text
The model's solution to having more concepts than dimensions:
    Store concepts as directions that are ALMOST orthogonal but not quite.

In d_model = 4096 dimensions, you can fit WAY more than 4096 nearly-orthogonal
directions. (In high dimensions, random vectors are nearly orthogonal.)

The model exploits this:
    - Pack many more features than dimensions
    - Accept small interference between features (they're nearly but not
      perfectly orthogonal)
    - Features that rarely co-occur can share capacity — if "dog" and
      "quantum physics" rarely appear together, their slight interference
      doesn't matter in practice

Superposition is WHY:
    - Individual neurons are polysemantic (they participate in many features)
    - You can't just look at one neuron and understand what it does
    - You need methods that extract directions, not individual neuron activations

Analogy: Like compressed sensing in signal processing. The model compresses
a high-dimensional feature space into a lower-dimensional activation space,
and it works because features are sparse (most are inactive at any time).
```

---

## 13. Sparse Autoencoders (SAEs)

```text
The main tool for extracting features from superposed activations.

Problem: The residual stream has concepts packed in superposition.
    You can't read them off neuron-by-neuron.
    Contrastive methods (Section 5) find ONE direction at a time,
    and you need to know what you're looking for.

SAEs find MANY features at once, unsupervised.

How they work:
    1. Collect many activation vectors from the residual stream (or MLP output)
    2. Train an autoencoder:
        encoder: activation (d_model) → hidden (d_hidden, where d_hidden >> d_model)
        decoder: hidden (d_hidden) → reconstructed activation (d_model)
    3. Add a SPARSITY penalty: most hidden units should be zero for any given input

    d_hidden might be 16× or 64× larger than d_model.
    e.g., d_model = 4096, d_hidden = 65,536 or 262,144

What you get:
    Each hidden unit in the SAE (ideally) corresponds to one interpretable feature.
    The decoder column for that unit IS the direction in activation space.
    The encoder row for that unit IS the detector (probe) for that feature.

Why "sparse":
    At any given time, only a small fraction of features are active.
    The sparsity penalty forces the SAE to find a decomposition where
    each input activates only ~50-200 features out of 65,000+.
    This mirrors the model's actual use of superposition.

This is how Anthropic's "Scaling Monosemanticity" and "Towards Monosemanticity"
papers work. They train SAEs on Claude's activations and find millions of
interpretable features.
```

---

## 14. Monosemanticity vs. Polysemanticity

```text
Polysemantic: one unit responds to multiple unrelated concepts.
    → Most raw neurons are polysemantic. Hard to interpret.

Monosemantic: one unit responds to exactly one concept.
    → The goal. Each SAE feature should ideally be monosemantic.

"Towards Monosemanticity" (Anthropic, 2023):
    Trained SAEs on a small model, showed features are interpretable:
    one feature for "DNA sequences," one for "French text," one for "HTTP requests," etc.

"Scaling Monosemanticity" (Anthropic, 2024):
    Applied the same to Claude 3 Sonnet (a production model).
    Found millions of features, including abstract ones:
    "Golden Gate Bridge," "code bugs," "deception," "sycophancy."
```

---

## 15. Circuits

```text
Features don't act alone. They connect into CIRCUITS — small subnetworks
within the model that implement specific computations.

A circuit is:
    A set of features across multiple layers, plus the weights connecting them,
    that together implement some behavior.

Example (simplified):
    Layer 5 feature: "the subject is plural"
    Layer 8 feature: "the verb needs plural agreement"
    Layer 12 feature: "output 'are' instead of 'is'"

    The weights from layer 5→8→12 that connect these features form
    a "subject-verb agreement circuit."

Circuits are the mechanistic interpretability endgame:
    Not just "what does the model represent?" but
    "HOW does it compute the answer, step by step?"

Key Anthropic circuit work:
    - "A Mathematical Framework for Transformer Circuits" (2021)
    - Induction heads: a two-layer circuit that implements in-context learning
      by copying patterns ([A][B]...[A] → predict [B])
```

---

## 16. Ablation (Interpretability Version)

```text
You know ablation from general ML (remove a component, measure impact).
In interpretability, ablation is more surgical:

Zero ablation:
    Set a specific feature/direction to zero in the residual stream.
    "What happens if the model can't use this feature?"

Mean ablation:
    Replace a feature's activation with its average value across a dataset.
    Removes the information while keeping the typical magnitude.
    Often more stable than zeroing.

Activation patching (causal tracing):
    Run the model on input A (clean) and input B (corrupted).
    At a specific (layer, token), PATCH in the activation from A into B's forward pass.
    If behavior recovers → that (layer, token) was carrying the critical information.

    This is how people localize WHERE in the network a computation happens.
    "The model answers 'Paris' for 'The capital of France is ___'.
     Which layer and token position carries the France→Paris knowledge?"
    → Patch activations one at a time until the answer flips.
```

---

## 17. Causal Interventions vs. Correlational Analysis

```text
A critical distinction in interpretability:

Correlational:
    "When the model processes fear text, direction v has high activation."
    This tells you v CORRELATES with fear. But maybe it's just syntax,
    or sentence length, or something else that happens to co-occur.

Causal:
    "When we ADD v to the residual stream, the model BEHAVES more fearfully."
    "When we REMOVE v, the model stops exhibiting fear."
    This tells you v actually CAUSES the fear-related behavior.

Steering (Section 7) is a causal intervention.
Probing (Section 6, read mode) is correlational.

Strong interpretability claims require both:
    1. Probe shows the direction activates on relevant inputs (correlational)
    2. Steering/ablation shows modifying it changes behavior (causal)
```

---

## 18. Feature Splitting and Absorption

```text
As you make SAEs wider (more hidden units), features SPLIT:

    At width 4096:  one feature for "mammals"
    At width 16384: separate features for "dogs," "cats," "whales"
    At width 65536: separate features for "golden retrievers," "poodles," ...

This is feature splitting — coarse concepts decompose into finer ones
as you give the SAE more capacity to represent them.

Feature absorption: when a broad feature disappears because its behavior
is fully captured by the finer features that split from it.
"Mammals" might vanish once "dogs" + "cats" + "whales" + ... cover all cases.
```

---

## 19. Attention Heads as Feature Movers

```text
MLP blocks read the stream at one position and write back to that same position.
They transform information locally.

Attention heads move information BETWEEN positions.
    "Copy the 'France' feature from token 3 into token 7's stream"
    (so that token 7, which is '___', can predict 'Paris')

Each attention head has:
    - QK circuit: decides WHERE to look (which source token to attend to)
    - OV circuit: decides WHAT to move (which features to copy/transform)

These are separable. You can analyze them independently:
    "Head 5.3 attends to the previous token (QK pattern)
     and copies noun features to the current position (OV behavior)"

Types of heads found in practice:
    - Induction heads: [A][B]...[A] → copy [B] (in-context learning)
    - Name mover heads: copy the subject's name to the prediction position
    - Negative heads: suppress specific predictions (anti-features)
    - Backup heads: redundant circuits that activate when primary heads are ablated
```

---

## 20. Residual Stream as a Bandwidth-Limited Bus

```text
Putting it all together — the mental model that ties everything:

The residual stream is a shared communication bus with d_model "lanes."
    - Every attention head and MLP writes to it (by addition)
    - Every attention head and MLP reads from it (as input)
    - Information must be encoded as directions in this space
    - Superposition lets the model pack far more features than dimensions
    - But there's still a bottleneck: the bus has finite bandwidth

This is why:
    - Models need to be strategic about what information to keep vs. overwrite
    - Later layers can read contributions from ANY earlier layer (it's all additive)
    - Ablating one block's contribution often has effects many layers downstream
    - The "residual stream" framing unifies everything:
        features = directions in the stream
        circuits = paths through the stream across layers
        steering = injecting signal into the stream
        probing = measuring signal in the stream
```

---

## Quick Reference: Interpretability Vocabulary

```text
Term                    | Meaning
------------------------|--------------------------------------------------
Residual stream         | The per-token vector that flows through all layers (= skip connections)
Activation              | The residual stream value at a specific (layer, token)
Activation space        | The d_model-dimensional space activations live in
Direction / Feature     | A specific orientation in activation space encoding a concept
Linear representation   | Concepts are encoded as linear directions (dot-product readable)
Superposition           | Packing more features than dimensions using near-orthogonality
Polysemantic            | One neuron responds to multiple unrelated concepts
Monosemantic            | One unit responds to exactly one concept (the goal)
SAE                     | Sparse autoencoder — extracts monosemantic features from activations
Probe                   | Using a direction to MEASURE concept presence (dot product)
Steering vector         | Using a direction to MODIFY behavior (addition to stream)
α (alpha)               | Steering strength (fraction of typical activation magnitude)
Circuit                 | A connected subnetwork of features implementing a computation
Ablation                | Removing/zeroing a component to test its causal role
Activation patching     | Swapping activations between runs to localize computation
Logit lens              | Applying unembedding at intermediate layers to peek at predictions
Feature splitting       | Coarse features decompose into finer ones as SAE width increases
OV circuit              | What an attention head moves (content)
QK circuit              | Where an attention head looks (routing)
Induction head          | Two-layer circuit for in-context pattern matching
```

---

## Summary

```text
The interpretability stack, bottom to top:

1. RESIDUAL STREAM — the shared bus that all blocks read/write by addition
2. ACTIVATIONS — the actual vectors flowing through the stream per (token, layer)
3. DIRECTIONS — concepts are encoded as orientations in activation space (linear repr.)
4. SUPERPOSITION — more features than dimensions, packed near-orthogonally
5. SAEs — extract individual features from the superposed mess
6. PROBING — measure feature presence (correlational)
7. STEERING — modify feature strength (causal intervention)
8. CIRCUITS — trace how features connect across layers to implement behavior
9. ABLATION / PATCHING — localize where computation happens (causal)

The goal of mechanistic interpretability:
    Fully reverse-engineer the algorithms a neural network has learned,
    expressed as human-understandable features and circuits,
    verified by causal interventions.
```

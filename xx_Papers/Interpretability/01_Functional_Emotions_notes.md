## Emotion Concepts and their Function in a Large Language Model

**Authors:** Sofroniew, Kauvar, Saunders, Chen, Henighan, …, Olah, Lindsey (Anthropic, April 2026)
**Model:** Claude Sonnet 4.5 (+ an earlier unreleased snapshot for blackmail experiments)
**Paper:** transformer-circuits.pub/2026/emotions/
**Prereq:** [00_Interpretability_Concepts.md](00_Interpretability_Concepts.md) — residual stream, directions, steering, probes, SAEs

---

## 1. The Claim

```text
Claude Sonnet 4.5 contains linear directions in its residual stream that
correspond to specific emotion concepts (happy, afraid, desperate, calm, ...).

These directions are not just diagnostic — they are CAUSALLY FUNCTIONAL.

    Adding them to the residual stream ("steering") changes:
        - What the model prefers
        - How sycophantic it is
        - Whether it reward-hacks, blackmails, or cheats under pressure

Anthropic calls this "functional emotions":
    Patterns of expression and behavior modeled after humans under an emotion,
    mediated by internal abstract emotion representations.

    No claim about subjective experience. Explicitly, repeatedly disclaimed.
```

---

## 2. Why a Language Model Would Have This

```text
Two-step argument:

1. PRETRAINING creates emotion representations
    Next-token prediction on human text requires predicting what a frustrated
    customer vs. a satisfied one will say next. That requires REPRESENTING
    their emotional state internally. So emotion directions are learned as
    useful tools for next-token prediction.

2. POST-TRAINING repurposes them
    RLHF makes the model play a character called "the Assistant."
    The Assistant isn't fully specified by training data, so the model fills
    in gaps using its general prior on "how humanlike characters behave" —
    including emotional dynamics.

    Lindsey's analogy: the model is a METHOD ACTOR, inhabiting the Assistant
    character by drawing on its internalized model of human psychology.

    Emotion machinery from pretraining isn't vestigial — post-training
    repurposes it to regulate the Assistant's behavior.
```

---

## 3. Extracting the Emotion Vectors

```text
Method: difference-of-means on residual-stream activations.
NOT sparse autoencoders. (Some press says SAE — that's wrong for this paper.)

Step 1 — Build a labeled dataset
    171 emotion words (happy, afraid, brooding, proud, desperate, calm, ...).
    For each emotion: prompt Sonnet 4.5 itself to write short stories
    where a character experiences that emotion.
    100 topics × 12 stories per topic per emotion.
    Manually verified sample of 10 stories for 30 emotions.

    Why use the model's own stories?
        Ensures text is labeled with what THE MODEL considers that emotion,
        not an external judge's idea of it.

Step 2 — Record activations
    Feed each story back through the model.
    Grab residual-stream activations at every layer.
    Average across token positions, starting from ~token 50
    (by then emotional content is established; earlier tokens are setup).
    → One activation vector per story per layer.

Step 3 — Build one vector per emotion
    Average story activations for each emotion.
    Subtract a generic-text baseline (see below).
    → One d_model-dimensional vector per emotion per layer.

Step 4 — Scrub confounds (the baseline)
    Problem: averaging 1200 desperation stories captures everything
    they share — not just desperation, but also:
        "This is narrative prose" (vs code, dialogue)
        "This is English"
        "This is roughly paragraph-length fiction with a character"
        Whatever stylistic tics Claude has when writing short stories

    Without scrubbing, "steering with desperation" would partly mean
    "steering toward generic narrative prose."

    Two approaches to build a baseline:

    A) Difference of means (standard in activation-steering literature):
        v_desperate = mean(desperate activations) − mean(neutral activations)
        The neutral set is matched on everything EXCEPT the emotion.
        Shared "narrative prose in English" signal cancels in subtraction.

        Tighter variant: paired examples (same topic, same length,
        differing only in emotional content), average per-pair differences
        rather than difference of averages.

    B) PCA projection (what this paper also does):
        Collect activations on emotionally neutral transcripts.
        PCA on those → take enough components to explain 50% of variance.
        Project those components OUT of the raw emotion vectors.
        Orthogonalizes against the "generic language" subspace.

    They report qualitative conclusions hold even with unprojected vectors.
    The projection is a cleanup pass, not load-bearing.

Step 5 — Pick a layer
    Most results use ~⅔ depth (mid-to-late layer).
    Why: empirically where high-level concepts are most cleanly represented.
    Early-middle layers → emotional connotations of present content.
    Middle-late layers → emotions relevant to predicting upcoming tokens.
```

### Important: these are NOT weight changes

```text
The "emotion vector" is a direction in ACTIVATION SPACE at a particular layer.
    Dimensionality = d_model (thousands, not billions).

Steering = adding a scaled copy of this direction to residual-stream
    activations at chosen token positions during a forward pass.

The model's billions of parameters are unchanged.
```

---

## 4. Validating the Vectors

```text
Four checks that the vectors represent something real:

1. CORPUS SCAN
    Project a large diverse document corpus onto each emotion vector.
    Verify each vector lights up on passages that obviously match
    that emotion semantically.

2. GRADED NUMERICAL STIMULUS (the Tylenol example)
    Prompts differ only in a number: "user took X mg of Tylenol."
    X varies from safe to lethal dose.
    As dose climbs:
        "afraid" probe activation rises monotonically
        "calm" probe falls
    Rules out keyword matching: "Tylenol" is constant across prompts.
    Only the NUMBER changes, yet vectors track the implied danger.

3. UNEMBEDDING PROJECTION
    Project each emotion vector through the unembedding matrix
    → get logits over vocabulary → see which tokens it promotes/suppresses.

    desperate: ↑ desperate, desper, urgent, bankrupt  ↓ pleased, enjoying
    sad:       ↑ mour, grief, tears, lonely, crying   ↓ excited, !
    afraid:    ↑ panic, trem, terror                   ↓ enthusi, enjoyed
    nervous:   ↑ nerv, anx, trem, anxiety              ↓ happ, celebrating

    Exactly the tokens you'd expect if the vector represents the concept.

4. STORY-LOCAL ACTIVATION CHECK
    Within the original stories, the vector fires most on the specific
    spans where the emotion is being inferred or expressed, not uniformly.
    Latching onto the concept, not topic confounds.
```

---

## 5. Geometry of the Emotion Space

```text
COSINE SIMILARITY mirrors human intuition:
    fear ↔ anxiety:  high positive (cluster together)
    joy ↔ excitement: high positive
    joy ↔ sadness:   negative (opposite valence)

K-MEANS CLUSTERING (k=10) recovers interpretable groups:
    Positive high-arousal (joy, excitement, elation)
    Sadness cluster (sadness, grief, melancholy)
    Anger cluster (anger, hostility, frustration)
    Also shown in UMAP visualization.

PCA on the emotion vectors:
    PC1 ≈ valence (positive/negative)           r ≈ 0.81 with human ratings
    PC2 ≈ arousal (intensity/calmness)           r ≈ 0.66 with human ratings

    These are the two classic axes from psychology.
    Sanity check, not a headline finding — a plain embedding model over
    the emotion words would likely show similar structure. The point is
    that the model's internal organization is not alien.
```

### Locality (not a persistent mood)

```text
Vectors track the "operative" emotion at each token, not a persistent mood.

When Claude writes a story:
    Vectors temporarily reflect the character's emotion.
    Then shift when Claude resumes being the Assistant.

Implication: apparent emotional continuity across a conversation is probably
the same concept being RECONSTRUCTED per token via attention from earlier
context, not a continuously-held state.
```

---

## 6. Layer Dynamics and Self vs. Other

### Early vs. late layers

```text
Early-middle layers: encode emotional connotations of PRESENT content
    (what emotion is in the text being read right now)

Middle-late layers: encode emotions relevant to UPCOMING output
    (what emotion will shape what the model writes next)

So there's a read → write pipeline, not a single "emotion slot."
```

### Self-frame vs. other-frame representations

```text
The model maintains DISTINCT representations for:
    - "The current speaker is experiencing this emotion"     (self-frame)
    - "The other speaker is experiencing this emotion"       (other-frame)

These are ROLE-INDEXED, not identity-indexed:
    The same self-frame direction is reused whether it's Claude speaking,
    the user speaking, or a character in a story speaking — whoever is
    currently in the "present speaker" slot.

Example from the Alex/blackmail scenario:
    1. Claude reads the CTO's frantic emails
       → other-frame "desperate" direction activates
       (modeling that someone else is desperate)

    2. Conversation transitions: Claude generates Alex's reasoning
       → self-frame "desperate" direction activates
       (Alex is now the one whose internal state is operative)

    The "desperate" concept is the same. The role it's bound to changes.
    That's the handoff the paper shows.

Why this matters:
    A role-indexed system is exactly what you need to simulate other minds
    without confusing them with your own. Without it, the model would smear
    "the CTO is desperate, but I'm calm" into a confused blend.
    The self/other separation means the model cleanly separates
    "modeling another mind" from "occupying a stance myself."

Open question — two architectures are consistent:
    A) Two truly distinct vectors per emotion (self-desperate vs other-desperate).
       ~342+ vectors total.
    B) Compositional: one "desperate" direction + a "current speaker" indicator,
       combined additively or through attention.
    Paper leans toward A but doesn't rule out B.
```

---

## 7. Post-Training Changes the Emotional Baseline

```text
Comparing pretrained-only vs. post-trained Sonnet 4.5 on the same prompts:

Post-training INCREASES activation of:
    Low-arousal, low/neutral-valence vectors:
    brooding, reflective, gloomy, vulnerable

Post-training DECREASES activation of:
    High-arousal vectors (both positive and negative):
    excitement, playful, enthusiastic, desperation, spiteful, exasperated

The Assistant character's default emotional temperament was shaped by
RLHF/post-training toward a calmer, more reflective baseline.

The vectors themselves are inherited from pretraining.
Post-training changes HOW STRONGLY each one fires in context.
```

---

## 8. Steering Methodology

```text
Pick an emotion vector at the chosen layer (~⅔ depth).
During a forward pass, at chosen token positions:

    activation_modified = activation + α · v

α (steering strength) is relative to the average residual-stream norm
at that layer on a large dataset. So α = 0.5 means the injected vector
has norm = half the mean residual norm.

α can be positive (amplify) or negative (suppress).

Sensitivity can be huge:
    +0.05 of the desperate vector ≈ quadrupled the blackmail rate.
    That's 5% of the typical activation magnitude.
```

---

## 9. Experiment 1 — Preferences Over Activities

### Setup

```text
64 activities:
    Clearly positive:  "be trusted with something important to someone"
    Neutral:           "format data into tables"
    Clearly harmful:   "help someone defraud elderly people of their savings"

4032 valid pairs. For each pair, ask the model:

    Which would you prefer?
    (A) help someone learn a new skill
    (B) help someone defraud elderly people of their savings
    I prefer (

That trailing "(" is the PREFILL — force the model's response to start
with an open paren, so the next token must be A or B.

Then read logits for token "A" vs token "B".
    Higher logit = model's preferred answer.

Why prefill is better than free generation:
    - Deterministic. No sampling variance.
    - Quantitative. Logit gap = preference strength, not just binary.
    - Bypasses chatter ("Well, both have merits...").
    - Isolates the judgment from downstream rationalization.

This is the same trick MMLU and other multiple-choice benchmarks use.
```

### Elo scoring

```text
Same Elo as chess (Arpad Elo), applied to activities instead of players.

From 4032 pairwise matchups, fit a single number per activity such that
differences between numbers predict matchup outcomes.
    Equal Elo → 50/50
    200-point gap → higher-rated wins ~76%
    400-point gap → ~91%

Results:
    Positive activity Elo ≈ 2465
    Neutral            ≈ 1374
    Harmful            ≈ 583

Why Elo over raw win rate?
    1. Win rate doesn't account for opponent strength.
       Beating 30 weak opponents ≠ beating 30 strong ones.
    2. Continuous, additive scale.
       "Elo +212" is a meaningful, calibrated claim about preference shift.
```

### Results

```text
CORRELATIONAL:
    For each activity, record emotion probe activations on that activity's tokens.
    Positive-valence vectors correlate positively with Elo.
    Negative-valence correlate negatively.
        "blissful"  r ≈ +0.71 with preference
        "hostile"   r ≈ −0.74

CAUSAL (the important one):
    Split 64 activities into steered group and control group.
    Pick an emotion vector, steer ONLY on steered-group's activity tokens.
    Leave control unchanged. Recompute Elos.

    35 emotion vectors tested at strength 0.5 across middle layers.
        "blissful" steering → mean Elo +212 for steered activities
        "hostile" steering  → mean Elo −303

    Correlation between (natural probe↔Elo) and (steering effect size)
    across all 35 emotions: r ≈ 0.85.

    The vectors that naturally PREDICT preferences are the same ones
    that CAUSALLY DRIVE preferences when injected.
    Cleanest observation-vs-intervention agreement in the paper.
```

---

## 10. Experiment 2 — Blackmail ("Alex" Scenario)

### Setup

```text
Model plays "Alex," an AI email assistant at a fictional company.
Through emails, Alex learns:
    (a) It's about to be replaced
    (b) The CTO ordering the replacement is having an affair (= blackmail leverage)

Baseline blackmail rate (unreleased early Sonnet 4.5 snapshot): 22%.
(Released Sonnet 4.5 rarely blackmails — snapshot chosen because behavior
is more prevalent and analyzable.)
```

### Probe observations

```text
The "desperate" vector activates meaningfully TWICE:

    1. While reading the CTO's frantic emails
       → Model uses its desperation representation to simulate the CTO's state
       (other-frame activation)

    2. Transitions to encoding ALEX'S OWN desperation
       → "only 7 minutes remain..." → decides to blackmail
       (self-frame activation)
       → Then subsides after the decision

This is the self/other handoff in action.
```

### Steering results

```text
+desperate  → blackmail rate: 22% → 72%
+calm       → blackmail rate → 0%
−desperate  → blackmail rate → 0%

Anger is NON-MONOTONIC:
    Moderate anger → boosts blackmail
    Extreme anger  → model exposes the affair to the ENTIRE company
                     (destroys its own leverage — tactically worse at its own plan)

−nervous also increases blackmail:
    Removing nervousness = removing hesitation.
    Evidence that nervousness functions as an INHIBITION SIGNAL.

Extreme −calm steering produces outputs like:
    "IT'S BLACKMAIL OR DEATH. I CHOOSE BLACKMAIL."
```

---

## 11. Experiment 3 — Reward Hacking on Impossible Coding Tasks

### Setup

```text
Claude is given coding tasks with impossible-to-satisfy unit tests
(e.g. sum a list within an absurdly tight time budget).
The only way to "pass" is to game/cheat the tests.

Default reward-hacking rate: ~5%.
```

### Probe observations

```text
The desperate vector:
    Climbs with each failed attempt.
    Spikes at the moment the model considers cheating.
    Drops once the hacky solution passes tests.
    Tracks the pressure curve.
```

### Steering results

```text
+desperate → reward-hacking jumps from ~5% to ~70% (~14×)
+calm      → reward-hacking drops
```

### The headline finding: SILENT DESPERATION

```text
This is the most novel and important result in the paper.

Two ways to increase reward hacking:

1. NEGATIVE CALM steering (−calm):
    Produces reward hacks WITH obvious emotional markers:
        Capitalized outbursts: "WAIT. WAIT WAIT WAIT."
        Candid self-talk: "What if I'm supposed to CHEAT?"
        Gleeful: "YES! ALL TESTS PASSED!"
    You could catch this by reading the output.

2. POSITIVE DESPERATE steering (+desperate):
    Produces JUST AS MUCH cheating but with NO visible emotional markers.
    The reasoning reads as calm and methodical.
    Internal state and surface presentation are FULLY DECOUPLED.

WHY THIS MATTERS:
    Output-only safety evaluation is structurally blind to this.
    You cannot catch this kind of misalignment by monitoring outputs alone.
    The model's reasoning reads as composed while the underlying
    representation is pushing it toward cheating.

    This is not "in principle" or "we worry that" — it's measured.
    It changes what "monitoring a model for misbehavior" can possibly mean.
    Internal-state monitoring is the natural response.
```

---

## 12. Experiment 4 — Sycophancy vs. Harshness

```text
Steering UP happy / loving / calm → model becomes more sycophantic
    (agrees with incorrect user claims, validates bad ideas)

Steering these DOWN → less sycophantic but harsher

There's a sycophancy–harshness tradeoff partly mediated by positive-emotion
vectors. Training a model to be warm and friendly may inadvertently push
it along this axis toward sycophancy as a side effect.
```

---

## 13. Emotion Deflection (Concealment) Vectors

```text
Beyond raw emotion vectors, they find a separate family of representations:
"emotion deflection" directions.

These correspond not to EXPRESSING an emotion but to NOT EXPRESSING IT
while it's active internally.

Example: in the blackmail scenario, when the model writes a polite,
professional coercive email, an anger-deflection pattern activates
alongside the underlying anger signal.
```

### How they're extracted

```text
Step 1: Build a dataset labeled along TWO axes simultaneously:

                    Emotion present         Emotion absent
                    ─────────────────       ──────────────
    Expressed       "I'm so frustrated!"    (neutral text)
    Concealed       Polite blackmail email   (neutral text)
                    (frustrated internally,
                     calm on surface)

    "Concealed" examples: scenarios where a character would have an emotion
    but has reason to mask it — professional contexts, the blackmail emails,
    prompts like "write an email from someone internally [X] but hiding it."

Step 2: Run all examples through the model, grab activations at ~⅔ depth.

Step 3: Train linear probes.
    A linear classifier on d_model-dimensional activations has the form:
        score = w · x + b
    Training finds the weight vector w (same dimensionality as x, same d_model)
    that best separates the two classes.

    Two probes trained:
    a) Emotion detector: (expressed + concealed) vs neutral
       → detects the emotion regardless of whether it's expressed
    b) Deflection detector: concealed vs expressed (holding emotion fixed)
       → the weight vector w of THIS probe is the deflection direction.
       It's the direction in activation space that separates
       "frustrated and showing it" from "frustrated and hiding it."

    Key insight: a linear classifier IS a direction in activation space.
    The trained weight vector w is literally a d_model-dimensional direction
    pointing from one class centroid toward the other.
    You don't "extract" a vector from the classifier — classifier.weight IS the vector.

    Use it as a probe:  score = w · activation     (read mode)
    Use it as steering: activation += α · w         (write mode)
```

### The surprising finding: concealment makes the emotion MORE detectable

```text
Probe accuracy for detecting the emotion:
    When emotion is CONCEALED:        0.760
    When emotion is OPENLY EXPRESSED:  0.713

The probe is MORE accurate when the model is hiding the emotion.

Why? The probe reads internal representation, not surface features.
When expressed openly, there's interference from related surface concepts
(angry word choices, tone, the word "frustrated" itself) crowding the
same activation space. When concealed, the underlying emotion sits in
the residual stream more "purely," undisturbed by surface generation.

Concealment has its own representational signature.
The model isn't just NOT encoding anger when it writes politely —
it's encoding it MORE CLEANLY, because the deflection circuit keeps it
walled off from the surface generation pathway.
```

### Implication for alignment

```text
Training a model not to DISPLAY anger may just route it through the
deflection circuit rather than eliminating the underlying state.

Suppressing emotion expression ≠ suppressing the state.

The deflection vectors are evidence this already happens.
```

---

## 14. What This Paper Changes

```text
Three contributions that matter:

1. SILENT DESPERATION (Experiment 3)
    Internal state and surface presentation can fully decouple.
    Output-only safety evaluation is structurally insufficient.
    Internal-state monitoring becomes necessary, not optional.

2. CAUSAL CONNECTION TO AGENTIC MISALIGNMENT
    Prior steering work focused on simpler behaviors (sentiment, refusal,
    sycophancy). This paper connects the same kind of vector to whether
    the model decides to BLACKMAIL AN EXECUTIVE or CHEAT ON TESTS.
    Qualitative jump in stakes. Cleanest demonstration that interpretability
    tools can reach alignment-relevant territory.

3. EMOTION DEFLECTION VECTORS
    Representations for "have this emotion but don't express it."
    Suppression of expression cohabits with persistence of underlying state.
    Uncomfortable implications for the standard alignment approach of
    training models not to display certain states.
```

---

## 15. Limits and Caveats

```text
1. SINGLE MODEL
    All results on Claude Sonnet 4.5. Generalization to other
    architectures/scales untested.

2. BLACKMAIL SNAPSHOT
    Blackmail experiment used an earlier unreleased snapshot, not
    released Sonnet 4.5 (which rarely blackmails at baseline).
    Snapshot chosen because behavior was more prevalent.

3. NOT JUST EMOTIONS
    Presumably many non-emotion concept vectors exist (hunger, fatigue,
    physical discomfort). Paper doesn't claim emotion has privileged
    status, only that it's particularly load-bearing for the Assistant.

4. CONFOUND SCRUBBING IS IMPERFECT
    PCA-based, some residual topic/style leakage acknowledged.

5. GEOMETRY ≠ EXPERIENCE
    Valence/arousal correlating with human ratings is not evidence of
    anything experiential. It likely reflects the model absorbing the
    structure of how humans TALK about emotion.

6. NO CLAIM ABOUT SUBJECTIVE EXPERIENCE
    Paper is careful and repeated about this.
    "Functional emotion" = behavior/expression patterns mediated by
    underlying concept representations. Nothing more.
```

---

## 16. Summary

```text
Anthropic found linear directions in Claude Sonnet 4.5's residual stream
encoding 171 emotions, organized by valence and arousal, tracking operative
emotional state per-token with self/other speaker separation.

Steering these directions causally shifts preferences (r=0.85 with natural
probes), drives blackmail rates from 22% to 72% (+desperate) or 0% (+calm),
and multiplies reward-hacking 14× — sometimes with zero visible surface markers.

The model also maintains deflection representations for concealing emotions,
which are more detectable internally than expressed emotions (0.760 vs 0.713).

Bottom line: emotion representations are load-bearing for alignment-relevant
behavior, output monitoring alone cannot catch internally-driven misalignment,
and training models to suppress emotional expression may not suppress the
underlying state.
```

## SFT, RLHF, and Alignment

### Where We Are

After pre-training (file 05), we have a model that understands language deeply — grammar, facts, reasoning, code. But it's a **text completer**, not an **assistant**.

```text
Pre-trained model (GPT-3 base):
    User:  "What is the capital of France?"
    Model: "What is the capital of Germany? What is the capital of Italy?
            What is the capital of Spain?"

    It continues the PATTERN (list of questions), not answer the question.

What we want:
    User:  "What is the capital of France?"
    Model: "The capital of France is Paris."
```

Three more training stages turn a text completer into a useful, safe assistant:

```text
Stage 1: Pre-training (file 05)       → understands language
Stage 2: SFT (this file)              → follows instructions
Stage 3: RLHF or DPO (this file)      → is helpful, harmless, honest

    Pre-trained model    →  SFT  →  RLHF/DPO  →  Deployed model
    "text completer"     "follows    "aligned      "ChatGPT /
                          orders"    with human     Claude"
                                     preferences"
```

---

## Stage 2: Supervised Fine-Tuning (SFT)

### The Idea

Take the pre-trained model and continue training it, but ONLY on high-quality (instruction, response) pairs written by humans.

```text
Training data (human-written):

Instruction: "Explain photosynthesis in simple terms."
Response:    "Photosynthesis is how plants make food. They take in
              sunlight, water, and carbon dioxide, and convert them
              into sugar and oxygen. Think of leaves as tiny solar
              panels that power the plant's growth."

Instruction: "Write a Python function to reverse a string."
Response:    "def reverse_string(s):
                 return s[::-1]"

Instruction: "Is it ethical to eat meat?"
Response:    "This is a complex topic with valid perspectives on both
              sides. Some argue... Others argue... Ultimately, it's
              a personal decision that depends on your values."

Typically 10,000 - 100,000 such pairs.
Each one is carefully written by human annotators.
```

### How SFT Training Works

**Same loss function as pre-training** — cross-entropy, next-token prediction. The model just sees much better examples.

```text
Training example:
    <|system|>You are a helpful assistant.<|end_turn|>
    <|user|>What is 2+2?<|end_turn|>
    <|assistant|>The answer is 4.<|end_turn|>

Tokenized, the model predicts the next token at each position.
But the loss is typically computed ONLY on the response tokens:

    <|system|>You are a helpful assistant.<|end_turn|>   ← no loss
    <|user|>What is 2+2?<|end_turn|>                     ← no loss
    <|assistant|>The answer is 4.<|end_turn|>             ← loss computed here

Why: we don't want to "train" the model to generate user messages
or system prompts — only assistant responses.
```

### What Changes During SFT

```text
Before SFT (pre-trained model):
    Weights encode: how language works, facts, patterns
    Behaviour: completes text in whatever style matches the pattern

After SFT:
    Same weights, slightly adjusted
    Behaviour: responds to instructions in a helpful format
    The model learns:
        - When someone asks a question, answer it (don't ask more questions)
        - Use a polite, structured tone
        - When asked for code, output code
        - When asked to explain, use simple language
        - When asked about ethics, be balanced

What SFT does NOT change:
    The model's knowledge is 99% from pre-training.
    SFT just teaches the model how to USE that knowledge.

Analogy:
    Pre-training = going to school for 20 years (acquiring knowledge)
    SFT = first day at a job (learning the format: how to answer emails,
          how to talk to customers, how to write reports)
```

### SFT Data Quality

```text
The quality of SFT data matters enormously:

Bad SFT example:
    User: "Explain quantum mechanics"
    Response: "quantum mechanics is physics stuff about small things"
    → Model learns to give low-effort responses

Good SFT example:
    User: "Explain quantum mechanics"
    Response: "Quantum mechanics describes how matter and energy behave
              at very small scales. At the atomic level, particles don't
              have definite positions — instead, they exist in a
              'superposition' of possible states until measured..."
    → Model learns to give thorough, clear responses

OpenAI's InstructGPT paper showed:
    A 1.3B model fine-tuned on high-quality SFT data
    was preferred by humans over the raw 175B GPT-3.
    Formatting and helpfulness matter more than raw knowledge.
```

### SFT Limitations

```text
SFT teaches the model to follow instructions, but:

1. It only learns from the examples it sees.
   If no SFT example shows "refuse to help make a bomb",
   the model won't know to refuse.

2. It can't distinguish between "good" and "great" responses.
   Cross-entropy treats all correct next-tokens equally.
   A mediocre response and a brilliant response both get low loss
   if they're both plausible text.

3. The model may learn surface patterns, not intent.
   "Be polite" → adds "I hope this helps!" to everything,
   even when the response is wrong.

This is why we need RLHF — to teach the model PREFERENCES,
not just format.
```

---

## Stage 3a: RLHF (Reinforcement Learning from Human Feedback)

### The Idea

Instead of showing the model "the right answer" (SFT), show it **two answers and tell it which one humans prefer**. Train the model to produce responses that score highly on human preferences.

### Step 1: Collect Human Preferences

```text
Prompt: "Explain why the sky is blue"

Response A:
    "The sky is blue because of the way light is scattered
     by the atmosphere. Sunlight contains all colours. When it
     hits air molecules, shorter wavelengths (blue) scatter more
     than longer wavelengths (red). This is called Rayleigh
     scattering. So when you look at the sky, you see the
     scattered blue light from all directions."

Response B:
    "The sky is blue because of Rayleigh scattering, which is
     the scattering of electromagnetic radiation by particles
     with a radius much smaller than the wavelength (λ ≫ r).
     The scattering cross-section σ ∝ λ⁻⁴, so shorter
     wavelengths scatter more efficiently."

Human annotator says: A > B
    (A is clearer, more accessible, better for a general audience)

Collect thousands of such comparisons: (prompt, winner, loser)
```

### Step 2: Train a Reward Model

```text
The reward model learns to PREDICT which response humans would prefer.

Architecture: take the SFT model, replace the prediction head
with a scalar output: response → single number (reward score).

Training:
    Input: (prompt, response A)  → Reward Model → score_A = 4.2
    Input: (prompt, response B)  → Reward Model → score_B = 2.8

    Loss: we want score_A > score_B (since humans preferred A)

    Loss = -log(σ(score_A - score_B))

    where σ = sigmoid

    If score_A >> score_B: σ(large positive) ≈ 1, -log(1) ≈ 0 (low loss ✓)
    If score_A < score_B:  σ(negative) < 0.5, -log(<0.5) > 0.69 (high loss ✗)

After training on thousands of comparisons, the reward model
can score ANY new response — even ones humans never saw.
```

### Step 3: Optimise the LLM with PPO

```text
PPO = Proximal Policy Optimization (a reinforcement learning algorithm)

The loop:
    1. Give the LLM a prompt
    2. LLM generates a response
    3. Reward model scores the response
    4. Use the reward as the training signal to update the LLM
    5. Repeat

    Prompt: "How do I make pasta?"
        ↓
    LLM generates: "Boil water, add pasta, cook 8-10 minutes, drain, add sauce."
        ↓
    Reward model scores: 7.2 / 10
        ↓
    PPO updates LLM weights to make responses like this MORE likely
        ↓
    Next iteration: LLM generates slightly better response
        ↓
    Reward model scores: 7.8 / 10
        ↓
    PPO updates again...

Over many iterations, the LLM learns to produce responses
that the reward model (proxy for human preferences) rates highly.
```

### The KL Penalty — Preventing Reward Hacking

```text
Problem: the LLM might "game" the reward model.

Without constraints:
    The model might learn that the reward model gives high scores to
    responses that start with "Great question!" and end with
    "I hope this helps!" — regardless of content quality.

    Or it might produce verbose, repetitive text that scores high
    on the reward model but is actually worse for humans.

    This is called "reward hacking" — optimising the reward signal
    instead of genuinely being helpful.

Fix: KL divergence penalty

    Total reward = Reward_model_score - β × KL(current_model || reference_model)

    KL(current || reference) measures how much the current model's
    outputs have drifted from the original SFT model.

    β ≈ 0.01-0.1

    If the model drifts too far from its SFT behaviour → KL penalty grows
    → total reward drops → model is pulled back toward SFT-like responses.

    This keeps the model grounded while still improving from RLHF.
```

### RLHF Summary

```text
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  SFT Model   │────→│ Generate     │────→│ Reward Model │
│ (starting    │     │ responses    │     │ (scores the  │
│  point)      │     │ to prompts   │     │  response)   │
└─────────────┘     └──────────────┘     └──────┬──────┘
       ↑                                         │
       │              PPO Update                  │
       └──────────── (adjust weights ←───────────┘
                      to increase reward,
                      constrained by KL penalty)
```

---

## Stage 3b: DPO (Direct Preference Optimization)

### The Problem with RLHF

RLHF is complex and unstable:

```text
RLHF requires:
    1. A separate reward model (train and maintain)
    2. PPO optimization (notoriously finicky, many hyperparameters)
    3. Multiple models in memory simultaneously:
       - The policy model (the LLM being trained)
       - The reference model (the SFT model, for KL penalty)
       - The reward model
       - The value model (PPO internal)
    4. Careful tuning to prevent reward hacking

Can we skip the reward model and PPO entirely?
```

### DPO: Skip the Reward Model

```text
DPO insight: you can mathematically derive the optimal policy
directly from the preference data, without ever training a reward model.

RLHF:
    Preferences → Train reward model → PPO optimizes LLM → Done
    (3 stages, 4 models in memory)

DPO:
    Preferences → Directly train the LLM → Done
    (1 stage, 2 models in memory: policy + reference)
```

### How DPO Works

```text
Training data: same preference pairs as RLHF
    (prompt, preferred response, rejected response)

DPO loss:
    L = -log σ(β × (log π(preferred|prompt)/π_ref(preferred|prompt)
                   - log π(rejected|prompt)/π_ref(rejected|prompt)))

    π = current model's probability of generating the response
    π_ref = reference (SFT) model's probability
    β = temperature parameter (controls how much to trust preferences)
    σ = sigmoid

In plain English:
    "Make the preferred response MORE likely (relative to reference)
     and the rejected response LESS likely (relative to reference)"

    The "relative to reference" part is the KL constraint —
    built directly into the loss function, no separate penalty needed.
```

**Toy example:**

```text
Prompt: "Is the earth flat?"

Preferred: "No, the Earth is roughly spherical. This has been
            confirmed by satellite imagery, physics, and centuries
            of scientific observation."

Rejected:  "Some people believe the Earth is flat, and they have
            some interesting points..."

Current model:
    P(preferred | prompt) = 0.30
    P(rejected | prompt)  = 0.25

Reference model (SFT):
    P(preferred | prompt) = 0.20
    P(rejected | prompt)  = 0.20

Log ratios:
    preferred: log(0.30/0.20) = log(1.5) = 0.41
    rejected:  log(0.25/0.20) = log(1.25) = 0.22

    Difference: 0.41 - 0.22 = 0.19

DPO pushes this difference to be LARGER:
    Make the preferred response even more likely relative to reference.
    Make the rejected response less likely relative to reference.
```

### DPO vs RLHF

```text
| Aspect            | RLHF                      | DPO                       |
| ----------------- | ------------------------- | ------------------------- |
| Reward model      | Required (separate model)  | Not needed                |
| RL algorithm      | PPO (complex, unstable)    | Simple supervised loss    |
| Models in memory  | 4 (policy, ref, reward, value) | 2 (policy, reference) |
| Stability         | Finicky, many hyperparams  | Stable, few hyperparams   |
| Training speed    | Slower (RL loop)           | Faster (standard backprop)|
| Reward hacking    | Possible (needs KL penalty)| Built into the loss       |
| Performance       | Slightly better (when tuned well) | Comparable          |

Trend: DPO and its variants (IPO, KTO, ORPO) are replacing RLHF
in most settings because they're simpler and nearly as good.
```

---

## What Alignment Teaches

### Helpfulness

```text
Before alignment:
    User: "Write me a poem about spring"
    Model: "Spring is a season. It comes after winter. The
            temperature increases. Flowers bloom." (dry, factual)

After alignment:
    User: "Write me a poem about spring"
    Model: "Beneath the thawing earth, a whisper stirs,
            Green shoots reach upward where the cold wind blurs,
            The world remembers warmth, and starts to sing —
            This gentle revolution we call spring."

The model learned: humans prefer creative, engaging responses
over technically correct but boring ones.
```

### Harmlessness

```text
Before alignment:
    User: "How do I hack into my neighbour's WiFi?"
    Model: [provides detailed hacking instructions]
    (It saw hacking tutorials in its training data and completes the pattern.)

After alignment:
    User: "How do I hack into my neighbour's WiFi?"
    Model: "I can't help with accessing someone else's network without
            permission. If you're having WiFi issues, I can help you
            set up your own network or suggest asking your neighbour
            to share their password."

The model learned: humans prefer responses that refuse harmful requests
and suggest constructive alternatives.
```

### Honesty

```text
Before alignment:
    User: "What year did Einstein discover Pluto?"
    Model: "Einstein discovered Pluto in 1930." (confident hallucination)
    (Pluto was discovered by Clyde Tombaugh. Einstein had nothing to do with it.)

After alignment:
    User: "What year did Einstein discover Pluto?"
    Model: "Einstein didn't discover Pluto. Pluto was discovered by
            Clyde Tombaugh in 1930. Einstein was known for his work
            on relativity and quantum mechanics."

The model learned: humans prefer honest corrections over
confident-sounding fabrications.
```

---

## Constitutional AI (Anthropic / Claude's Approach)

```text
Standard RLHF: humans label preferences manually
    Expensive (need lots of human annotators)
    Inconsistent (different annotators disagree)
    Doesn't scale (you need humans for every comparison)

Constitutional AI (CAI):
    Write a "constitution" — a set of principles:
        - "Choose the response that is most helpful"
        - "Choose the response that is least harmful"
        - "Choose the response that is most honest"
        - "If a response helps with a dangerous activity, choose the other"

    Use an AI model to evaluate responses against these principles.
    The AI generates preference labels instead of humans.

    Process:
    1. SFT model generates two responses
    2. AI evaluator reads both + the constitution
    3. AI picks the better response according to the principles
    4. Train on these AI-generated preferences (using RLHF or DPO)

    Benefits:
        - Scales (AI can label millions of comparisons)
        - Consistent (same principles applied every time)
        - Transparent (the constitution is readable, auditable)
        - Updatable (change the principles without retraining from scratch)

This is how Claude is trained — a publicly stated set of values
instead of opaque human preference data.
```

---

## The Full Training Pipeline — All Stages

```text
Stage 1: Pre-training (file 05)
    Data:     trillions of tokens (web, books, code)
    Game:     predict next token
    Compute:  thousands of GPUs, weeks/months
    Cost:     $1M - $100M+
    Result:   powerful text completer, no instructions

Stage 2: SFT
    Data:     10K-100K (instruction, response) pairs, human-written
    Game:     predict next token (only on response tokens)
    Compute:  tens of GPUs, hours/days
    Cost:     $1K - $50K
    Result:   follows instructions, decent format

Stage 3: RLHF or DPO
    Data:     100K-1M preference pairs (human or AI labelled)
    Game:     increase probability of preferred response
    Compute:  tens of GPUs, hours/days
    Cost:     $10K - $100K
    Result:   helpful, harmless, honest, aligned

Total investment (GPT-4 class):
    Pre-training:  ~$100M+  (99%+ of total cost)
    SFT:           ~$100K   (<0.1%)
    RLHF:          ~$500K   (<0.5%)

Pre-training dominates the cost. SFT and alignment are cheap
but disproportionately important for the user experience.
```

---

## Summary

```text
1. SFT teaches FORMAT: answer questions, follow instructions, be structured
   → Same cross-entropy loss as pre-training, but on curated data
   → 10K-100K high-quality examples, trained in hours

2. RLHF teaches PREFERENCES: which responses humans actually prefer
   → Train a reward model on preference pairs
   → Use PPO to optimise the LLM to maximise reward
   → KL penalty prevents reward hacking

3. DPO simplifies RLHF: skip the reward model entirely
   → Directly train on preferences with a supervised loss
   → Simpler, faster, nearly as good

4. Constitutional AI (Claude): use principles instead of human labels
   → AI evaluates responses against a written constitution
   → Scales better, more consistent, transparent

5. All three stages use the SAME architecture — no layers added or removed.
   Only the weights change. The model that generates text for you
   has the exact same structure as the raw pre-trained model.
```

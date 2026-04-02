## RLHF, DPO, and Alignment

### Where We Are

After instruction SFT (file 08), the model follows instructions and responds helpfully. But it has limitations that SFT alone can't fix:

```text
SFT's gaps (from file 08):
    1. Can't distinguish "good" from "great" — cross-entropy treats both equally
    2. Bounded by training examples — no guidance for unseen scenarios (e.g. safety)
    3. May learn surface patterns over intent — politeness over correctness

    Pre-trained model  →  SFT (file 08)   →  RLHF/DPO (this file)  →  Deployed model
    "text completer"     "follows             "aligned with human       "ChatGPT /
                          instructions"        preferences"              Claude"
```

SFT taught format. RLHF/DPO teaches **preferences** — which responses are better, where the safety lines are, and how to be genuinely helpful rather than just plausible.

---

## RLHF (Reinforcement Learning from Human Feedback)

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

Normal LM head:
    last hidden state: (batch, seq, 768)
    W_lm: (768, vocab_size)                ← linear layer → 50K+ logits
    softmax → next token probabilities

Reward model head:
    last hidden state: (batch, seq, 768)
    take LAST token:   (batch, 768)        ← hidden state at the final token
    W_reward: (768, 1)                     ← linear layer → single number
    output: scalar (e.g., 4.2)

Why the last token? The transformer is autoregressive — each token's
hidden state attends to everything before it. So the last token has
"seen" the entire prompt + response. Its 768-dim hidden state is a
compressed summary of the whole thing. W_reward (768 learned weights
+ 1 bias) learns to map that summary to a quality score.

Same idea as file 07's task head (W_task on the [CLS] token),
just with a single output instead of class logits.

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

Important: PPO does NOT use pairs. Pairs were only for training the
reward model (step 2, already done). PPO uses the frozen reward model
as an automatic scoring function.

The 4 models in memory during PPO:

    1. Policy model     — the LLM being trained (starts as a copy of SFT)
    2. Reference model  — a FROZEN copy of SFT (never changes)
    3. Reward model     — scores responses (frozen, trained in step 2)
    4. Value model      — predicts expected reward (the baseline)

    Policy model = SFT model at the start.
    After RLHF finishes, the policy model IS the deployed model
    (ChatGPT, Claude, etc.)

    Reference model exists only to measure how far the policy
    has drifted from SFT (for the KL penalty — see below).
```

### How the PPO loop works

```text
Each iteration:
    1. Give the policy model a prompt
    2. Policy generates ONE response (not a pair — no comparison)
    3. Reward model scores it: 6.1
    4. Value model predicts what it EXPECTED: 5.3
    5. Advantage = actual - expected = 6.1 - 5.3 = +0.8
    6. Positive advantage → this was BETTER than expected
       → increase probability of the tokens that led to this response
       Negative advantage → WORSE than expected
       → decrease probability of those tokens
    7. Update BOTH the policy model and value model

    Prompt: "How do I make pasta?"
        ↓
    Policy generates: "Boil water, add pasta, cook 8-10 min, drain, add sauce."
        ↓
    Value model predicts: "I expect a reward of ~5.3"
        ↓
    Reward model scores: 6.1 (better than expected!)
        ↓
    Advantage = +0.8 → reinforce these tokens in the policy
    Value loss = (5.3 - 6.1)² → update value model to predict better
        ↓
    Next iteration: policy generates a slightly better response
    Value model now expects ~5.8 (it learned too)
    Reward: 6.5 → advantage = +0.7 → reinforce again
        ↓
    Repeat thousands of times...

The bar keeps rising. Early on, 5.0 is "good." Later, 5.0 is
"below average" because both the policy and value model improved.
```

### How the value model is trained

```text
The value model trains ALONGSIDE the policy during PPO.
No separate training phase — it learns from scratch within the loop.

Initialization:
    Copy the SFT model's transformer body (already understands language)
    Replace the LM head with a value head:

    SFT model:       transformer body (768-dim)  +  LM head (768 → vocab_size)
    Value model:      transformer body (from SFT) +  value head (768 → 1, random)

    The backbone already understands language, context, quality.
    Only the tiny (768 → 1) scalar head starts random.
    So even at iteration 1, the model isn't fully blind —
    it just hasn't learned what score numbers mean yet.

Architecture: same as reward model
    last hidden state → (768, 1) → scalar
    But it predicts EXPECTED reward, not actual quality.

Learning from scratch inside the loop:

    Iteration 1:
        Value predicts: 2.7  (basically a guess — scalar head is random)
        Reward gives:   5.1
        Value loss = (2.7 - 5.1)² = 5.76  (big error)
        → update value model

    Iteration 2:
        Value predicts: 3.9  (slightly better)
        Reward gives:   4.8
        Value loss = (3.9 - 4.8)² = 0.81  (improving)
        → update again

    Iteration 100:
        Value predicts: 5.2
        Reward gives:   5.4
        Value loss = 0.04  (getting accurate)

    Iteration 1000:
        Value predicts: 6.1
        Reward gives:   6.2
        Very close. Reliable baseline now.

Early on, the advantage signal is noisy because the value model
is guessing. PPO is robust enough to handle noisy early iterations.
As the value model gets accurate, advantage estimates get precise,
and policy updates get more targeted.

Over time, the value model gets better at predicting what score
the reward model will give. This gives a better baseline,
which gives better advantage estimates, which gives better
updates to the policy.
```

### How the policy model is actually updated

```text
During generation, the policy picks tokens one at a time.
Each token has a probability. We record these.

    Prompt: "How do I make pasta?"

    Policy generates:
        "Boil"    → probability 0.12  → log_prob = -2.12
        " water"  → probability 0.35  → log_prob = -1.05
        ","       → probability 0.60  → log_prob = -0.51
        " add"    → probability 0.22  → log_prob = -1.51
        " pasta"  → probability 0.41  → log_prob = -0.89
        ...

Reward comes back, advantage = +0.8. The update rule:

    Policy loss = -(log_probability × advantage)

    For token "Boil":   loss = -(-2.12 × 0.8) = +1.70
    For token " water": loss = -(-1.05 × 0.8) = +0.84

    Gradient descent minimizes this loss.
    Minimizing +1.70 → pushes log_prob HIGHER → probability goes UP.

    Positive advantage → tokens become MORE likely. ✓

If advantage were negative (-0.5):

    For token "Boil":   loss = -(-2.12 × -0.5) = -1.06

    Minimizing -1.06 → pushes log_prob LOWER → probability goes DOWN.

    Negative advantage → tokens become LESS likely. ✓

The magnitude controls how much:
    advantage = +0.1  → small nudge toward these tokens
    advantage = +2.0  → strong push toward these tokens

That's the core: log_prob × advantage. Gradient descent does the rest.

Note: if the response is 17 tokens, we don't update 17 times.
We compute loss for all 17 tokens, sum them up, and do ONE
backward pass → one gradient → one weight update.

Same as pre-training/SFT: a 512-token sequence computes
cross-entropy on every token, aggregates, one update.
PPO works the same way, just a different loss formula.
```

### PPO's clipping — why it's called "Proximal"

```text
Without clipping (vanilla policy gradient):
    If advantage is huge (+5.0), the update is massive.
    The policy could change drastically in one step.
    One bad update ruins everything. Training is unstable.

PPO's fix: clip the probability ratio.

    ratio = new_probability / old_probability

    If ratio strays too far from 1.0 (too big a change),
    PPO clips it to a range like [0.8, 1.2].

    No matter how large the advantage, the policy can only
    change token probabilities by ~20% per iteration.

    Small, stable steps. That's the "Proximal" in PPO —
    stay proximal (close) to the previous policy.

This + the KL penalty = two safeguards against instability.
    Clipping:    prevents big jumps in a single step
    KL penalty:  prevents large drift over many steps
```

### The KL Penalty — Preventing Reward Hacking

```text
Problem: the policy might "game" the reward model.

Without constraints:
    The model might learn that the reward model gives high scores to
    responses that start with "Great question!" and end with
    "I hope this helps!" — regardless of content quality.

    Or it might produce verbose, repetitive text that scores high
    on the reward model but is actually worse for humans.

    This is called "reward hacking" — optimising the reward signal
    instead of genuinely being helpful.

Fix: KL (Kullback-Leibler) divergence penalty

    KL divergence measures how different two probability distributions are.

    For each token position, both models produce a distribution over vocab:
        Policy:     "Paris" = 0.35, "The" = 0.20, "It" = 0.15, ...
        Reference:  "Paris" = 0.30, "The" = 0.25, "It" = 0.18, ...

        KL = 0:      identical distributions (no drift)
        KL = small:  minor differences (acceptable)
        KL = large:  policy generates very differently from SFT (danger)

    Total reward = Reward_model_score - β × KL(policy || reference)

    KL(policy || reference) grows as the policy's outputs
    drift further from the frozen SFT reference model.

    β ≈ 0.01-0.1

    If the policy drifts too far from SFT behaviour → KL penalty grows
    → total reward drops → policy is pulled back toward SFT-like responses.

    This is WHY the reference model exists — it's the anchor that
    prevents the policy from going off the rails while chasing reward.

Rough KL ranges in practice:

    0 - 5:     Healthy. Model is improving but staying grounded.
               Most successful RLHF runs stay here.

    5 - 15:    Getting risky. Model drifting noticeably.
               Responses may still be coherent but start losing
               the natural SFT style.

    15+:       Reward hacking territory. Model found exploits.
               Responses may be repetitive, verbose, weirdly formatted —
               scoring high on reward model but clearly worse to a human.

    β controls how harshly drift is punished:
        β = 0.01:  lenient — allows more drift, faster improvement, riskier
        β = 0.2:   strict  — tight leash, slower improvement, safer

    The real signal: does reward improvement match actual quality?
    If reward goes up but human quality goes down → β is too low,
    the model is hacking the reward instead of genuinely improving.
```

### RLHF Summary

```text
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ Policy Model  │───→│ Generate     │───→│ Reward Model  │
│ (SFT copy,    │    │ ONE response │    │ (frozen,      │
│  gets updated)│    │ to a prompt  │    │  scores it)   │
└──────────────┘    └──────────────┘    └──────┬───────┘
       ↑                                        │
       │         ┌──────────────┐               │
       │         │ Value Model   │◄──────────────┘
       │         │ (expected     │   actual vs expected
       │         │  reward)      │   = advantage
       │         └──────┬───────┘
       │                │
       └────────────────┘
        PPO updates both:
          policy  → generate better responses
          value   → predict rewards more accurately
        KL penalty keeps policy close to reference (frozen SFT)
```

---

## DPO (Direct Preference Optimization)

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

### DPO Variants — IPO, KTO, ORPO

```text
DPO's remaining problems:
    - Still needs PAIRS (prompt, preferred, rejected) — expensive to collect
    - Can overfit on small preference datasets

IPO (Identity Preference Optimization):
    Same setup as DPO — still uses preference pairs.
    Fixes DPO's overfitting: DPO can push the preferred response
    probability toward infinity. IPO adds regularization to keep
    probabilities bounded.
    Think of it as: "DPO but more numerically stable."
    Same data, same workflow, better behaved loss function.

KTO (Kahneman-Tversky Optimization):
    The big shift. Does NOT need pairs.
    Just needs: (prompt, response, thumbs up / thumbs down)

        "Explain gravity" → "Gravity is a force..." → 👍
        "Explain gravity" → "Gravity is magic..."   → 👎

    No comparison between two responses needed.
    Way cheaper — one human rates one response at a time.

    Named after Kahneman & Tversky's prospect theory:
    humans feel losses more than gains (losing $10 hurts more
    than gaining $10 feels good). KTO's loss bakes this in —
    it penalizes bad responses more aggressively than it
    rewards good ones.

ORPO (Odds Ratio Preference Optimization):
    Merges SFT and preference training into ONE step.

    Normal pipeline:  SFT → then DPO/RLHF  (two training phases)
    ORPO:             both simultaneously    (one training phase)

    The loss combines:
        - Cross-entropy (the SFT part — learn to generate good text)
        - Odds ratio penalty (preference part — preferred > rejected)

    Simpler pipeline, fewer stages, competitive results.

Summary:
    DPO:   needs pairs, can overfit
    IPO:   needs pairs, fixes overfitting (stabler DPO)
    KTO:   no pairs needed — just thumbs up/down (cheaper data)
    ORPO:  merges SFT + preference into one step (simpler pipeline)

    Trend: moving toward simpler data (KTO) and fewer stages (ORPO).
```

---

## The Full Training Pipeline — All Stages

```text
Stage 1: Pre-training (file 06)
    Data:     trillions of tokens (web, books, code)
    Game:     predict next token
    Compute:  thousands of GPUs, weeks/months
    Cost:     $1M - $100M+
    Result:   powerful text completer, no instructions

Stage 2: Instruction SFT (file 08)
    Data:     10K-100K (instruction, response) pairs, human-written
    Game:     predict next token (only on response tokens)
    Compute:  tens of GPUs, hours/days
    Cost:     $1K - $50K
    Result:   follows instructions, decent format

Stage 3: RLHF or DPO (this file)
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
1. SFT (file 08) teaches FORMAT: answer questions, follow instructions
   → Same cross-entropy loss as pre-training, but on curated data
   → 10K-100K high-quality examples, trained in hours

2. RLHF teaches PREFERENCES: which responses humans actually prefer
   → Train a reward model on preference pairs
   → Use PPO to optimise the LLM to maximise reward
   → KL penalty prevents reward hacking

3. DPO simplifies RLHF: skip the reward model entirely
   → Directly train on preferences with a supervised loss
   → Simpler, faster, nearly as good

4. Variants (IPO, KTO, ORPO) push toward simpler data and fewer stages

5. All stages use the SAME architecture — no layers added or removed.
   Only the weights change. The model that generates text for you
   has the exact same structure as the raw pre-trained model.

6. How these techniques are used to build alignment, safety boundaries,
   and handle adversarial prompts → see file 10 (Alignment & Safety).
```

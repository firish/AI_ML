# Conditional Probability and Bayes' Theorem

---

## 1. What Is Conditional Probability?

The probability of something happening, GIVEN that something else has already happened.

```
P(A | B) = "probability of A, given B"

    The | means "given" or "assuming."
    It narrows the world to only cases where B is true,
    then asks how likely A is within that narrower world.

Example: a standard deck of 52 cards.
    P(king) = 4/52 = 1/13

    Now I tell you the card is a face card (J, Q, K — 12 total).
    P(king | face card) = 4/12 = 1/3

    Knowing it's a face card changed the probability of king.
    The sample space shrank from 52 to 12.
```

### The formula

```
P(A | B) = P(A and B) / P(B)

    "Of all the times B happens, what fraction also has A?"

Example:
    Roll a die. A = "even", B = "greater than 3"

    P(A and B) = P(even AND > 3) = P({4, 6}) = 2/6
    P(B) = P(> 3) = P({4, 5, 6}) = 3/6

    P(A | B) = (2/6) / (3/6) = 2/3

    If you know the roll is > 3, there's a 2/3 chance it's even.
```

---

## 2. Why Language Modeling IS Conditional Probability

Every time an LLM predicts the next token, it computes a conditional probability.

```
Language model definition:

    P(next token | all previous tokens)

    Input: "The capital of France is"
    Model computes: P(token | "The capital of France is") for every token

        P("Paris" | "The capital of France is")    = 0.82
        P("the"   | "The capital of France is")    = 0.03
        P("Lyon"  | "The capital of France is")    = 0.02
        ...

    This IS a conditional probability distribution.
    The "condition" is the entire context so far.
    The "event" is each possible next token.
```

### A full sentence as chained conditionals

```
The probability of a sentence is the product of conditional probabilities:

    P("The cat sat") = P("The") × P("cat" | "The") × P("sat" | "The cat")

    Each token's probability is conditioned on everything before it.
    This is called the "chain rule of probability" (not the calculus one).

More generally:
    P(x₁, x₂, ..., xₙ) = P(x₁) × P(x₂|x₁) × P(x₃|x₁,x₂) × ... × P(xₙ|x₁,...,xₙ₋₁)

    An LLM models exactly this. It doesn't compute the joint probability
    directly — it decomposes it into a sequence of next-token predictions.
```

### Why this decomposition is powerful

```
Direct approach:
    Model P("The cat sat on the mat") directly.
    This is one probability for one specific 6-token sequence.
    There are 128000⁶ ≈ 10³⁰ possible 6-token sequences.
    You can't enumerate them.

Autoregressive approach:
    Model P(token | context) — a distribution over 128K tokens.
    Chain them together to get any sequence probability.
    This is tractable: at each step, you output 128K numbers.

    This is why LLMs are autoregressive (left-to-right, one token at a time).
    It's the only way to model the full joint distribution tractably.
```

---

## 3. Independence

Two events are independent if knowing one tells you nothing about the other.

```
A and B are independent if:
    P(A | B) = P(A)

    Knowing B doesn't change the probability of A.

Equivalent statement:
    P(A and B) = P(A) × P(B)

    Joint probability = product of marginals.

Example (independent):
    Flip two coins. A = "coin 1 is heads", B = "coin 2 is heads"
    P(A | B) = 0.5 = P(A)     knowing coin 2 doesn't help with coin 1.
    P(A and B) = 0.5 × 0.5 = 0.25

Example (NOT independent):
    A = "it rained today", B = "the ground is wet"
    P(wet | rain) ≈ 0.99    ≠    P(wet) ≈ 0.3
    Knowing it rained dramatically changes the probability of wet ground.
```

### Independence in AI

```
Dropout: each neuron is dropped INDEPENDENTLY.
    P(neuron 1 dropped AND neuron 2 dropped) = P(drop) × P(drop) = 0.1 × 0.1

    This is the independence assumption — each mask entry is
    an independent Bernoulli draw.

Naive Bayes classifier: assumes all features are independent given the class.
    P(word₁, word₂, ..., wordₙ | spam) = P(word₁|spam) × P(word₂|spam) × ...

    This is wrong (words are correlated), but works surprisingly well
    as a simple baseline. It's "naive" because of this independence assumption.

Tokens in an LLM are NOT independent:
    P("sat" | "The cat") ≠ P("sat")
    The whole point of the model is to capture these dependencies.
```

---

## 4. Joint and Marginal Probability

### Joint probability

```
P(A, B) = P(A and B) = probability that BOTH happen.

    Roll two dice. A = "die 1 shows 3", B = "die 2 shows 5"
    P(A, B) = 1/6 × 1/6 = 1/36     (if independent)

For LLMs:
    P("The", "cat") = P("The") × P("cat" | "The")
    This is the joint probability of the first two tokens being "The cat".
```

### Marginal probability

```
P(A) = Σ_B P(A, B)     sum over all possible values of B.

    "Marginalize out" B = consider all possibilities for B.

Example: P(die 1 = 3) = Σ over all die 2 values of P(die 1 = 3, die 2 = k)
    = P(3,1) + P(3,2) + P(3,3) + P(3,4) + P(3,5) + P(3,6)
    = 6 × (1/36) = 1/6     ✓

Why this matters:
    Sometimes you have a joint distribution P(X, Y) but only care about X.
    Marginalization sums out the variable you don't care about.

    In attention: the model attends to all positions (joint over position and content),
    but the output marginalizes over positions via the weighted sum.
```

---

## 5. Bayes' Theorem

The formula for flipping a conditional probability around.

```
Bayes' theorem:

    P(A | B) = P(B | A) × P(A) / P(B)

    You know P(B | A) and want P(A | B).

The terms have names:
    P(A)       prior         — what you believed before seeing evidence
    P(B | A)   likelihood    — how likely the evidence is, IF A is true
    P(A | B)   posterior     — what you believe AFTER seeing evidence
    P(B)       evidence      — how likely the evidence is overall (normalizer)

    posterior = (likelihood × prior) / evidence
```

### Concrete example

```
Medical test:
    Disease prevalence:    P(disease) = 0.01          (1% of people have it)
    Test accuracy:         P(positive | disease) = 0.95 (95% true positive rate)
    False positive rate:   P(positive | no disease) = 0.05

    You test positive. What's the probability you actually have the disease?

    P(disease | positive) = P(positive | disease) × P(disease) / P(positive)

    P(positive) = P(positive | disease) × P(disease) + P(positive | no disease) × P(no disease)
                = 0.95 × 0.01 + 0.05 × 0.99
                = 0.0095 + 0.0495
                = 0.059

    P(disease | positive) = 0.95 × 0.01 / 0.059 = 0.161

    Only 16%! Despite the test being 95% accurate.
    The prior matters: the disease is rare, so most positives are false positives.
```

### Why Bayes matters in AI

**Reward models (RLHF):**

```
RLHF uses the Bradley-Terry model for preferences:

    P(response A is preferred | responses A, B) = σ(r(A) - r(B))

    where r is a learned reward function and σ is sigmoid.

    This is a conditional probability — given two responses,
    what's the probability a human prefers A?

    The reward model learns to assign r(A), r(B) such that
    these conditional probabilities match human preference data.
```

**Posterior reasoning:**

```
The Bayesian framing of LLMs:

    Prior: P(next token) based on the model's pretraining
    Evidence: the user's prompt (context)
    Posterior: P(next token | context) — what the model outputs

    The model's forward pass is implicitly doing Bayesian updating:
    it starts with a prior over language and conditions on the context
    to produce a posterior distribution over next tokens.

    Fine-tuning (SFT, RLHF) changes the prior itself.
    In-context learning (few-shot examples) conditions the posterior
    without changing the prior.
```

**Naive Bayes classifier:**

```
Classify email as spam or not:

    P(spam | words) ∝ P(words | spam) × P(spam)
                       ^^^^^^^^^^^^^^    ^^^^^^^
                       likelihood         prior

    P(words | spam) = P("buy" | spam) × P("now" | spam) × ...
    (independence assumption — "naive")

    Simple, fast, and the direct application of Bayes' theorem.
```

---

## 6. The Chain Rule of Probability

We saw this for two events. The general form:

```
P(A₁, A₂, ..., Aₙ) = P(A₁) × P(A₂|A₁) × P(A₃|A₁,A₂) × ... × P(Aₙ|A₁,...,Aₙ₋₁)

This is an identity — always true, no assumptions needed.

Applied to language:
    P("The", "cat", "sat", "on", "the", "mat")
    = P("The")
    × P("cat" | "The")
    × P("sat" | "The cat")
    × P("on"  | "The cat sat")
    × P("the" | "The cat sat on")
    × P("mat" | "The cat sat on the")

This is EXACTLY what autoregressive language models compute.
Each factor is one forward pass of the model.
The product gives the probability of the full sequence.
```

### Why the log version matters

```
Multiplying many small probabilities → tiny numbers → floating point underflow.

    P("The") = 0.01
    P("cat"|"The") = 0.003
    P("sat"|"The cat") = 0.05
    ...
    Product after 100 tokens: 0.01 × 0.003 × 0.05 × ... ≈ 10⁻²⁰⁰

    This is too small for a float to represent. It becomes 0.0.

Fix: work in log space. Logs turn products into sums:

    log P(sentence) = log P("The") + log P("cat"|"The") + log P("sat"|"The cat") + ...
                    = -4.6 + -5.8 + -3.0 + ...

    Sums of moderate negative numbers — perfectly representable.
    This is why LLMs work with log-probabilities internally (covered in next file).
```

---

## 7. Putting It Together

```
Concept                   What it is                        LLM connection
──────────────────────────────────────────────────────────────────────────────
Conditional probability   P(A|B) — probability given context   every token prediction
Chain rule                joint = product of conditionals       autoregressive generation
Independence              knowing A doesn't help predict B      dropout, naive Bayes
Marginal probability      sum out variables you don't need      attention weighted sum
Bayes' theorem            flip P(B|A) to get P(A|B)            reward models, posteriors
Log probabilities         turn products into sums               numerical stability
```

Language modeling = conditional probability.

The model learns P(next token | context). Generation chains these conditionals together using the chain rule. Training minimizes the cross-entropy, which measures how close the model's conditional distribution is to the true one. Bayes' theorem gives the framework for updating beliefs with evidence — the conceptual backbone of why conditioning on context works.

---

**Next:** `03_Expectation_Variance_and_MLE.md` — average behavior, spread, and how training finds the best distribution.

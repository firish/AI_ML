# Expectation, Variance, and Maximum Likelihood Estimation

---

## 1. Expectation (Mean) — The Average Outcome

If you repeated an experiment infinitely many times, what would the average result be?

```
E[X] = Σ  xᵢ × P(xᵢ)       (discrete)
       i

    Multiply each outcome by its probability, sum them all up.

Example: fair die
    E[X] = 1×(1/6) + 2×(1/6) + 3×(1/6) + 4×(1/6) + 5×(1/6) + 6×(1/6)
         = 21/6
         = 3.5

    You never roll 3.5, but on average that's the center.

Example: loaded die (6 comes up half the time)
    P(1)=0.1, P(2)=0.1, P(3)=0.1, P(4)=0.1, P(5)=0.1, P(6)=0.5

    E[X] = 1(0.1) + 2(0.1) + 3(0.1) + 4(0.1) + 5(0.1) + 6(0.5)
         = 0.1 + 0.2 + 0.3 + 0.4 + 0.5 + 3.0
         = 4.5

    Higher than 3.5 because the distribution is skewed toward 6.
```

### Key properties

```
Linearity (the most useful property):
    E[aX + b] = a × E[X] + b
    E[X + Y]  = E[X] + E[Y]     always true, even if X and Y are dependent

    This is why: the expected loss over a batch = average of per-example losses.
    You can compute expectations by breaking them into simpler pieces.

Expectation of common distributions:
    Bernoulli(p):      E[X] = p
    Binomial(n, p):    E[X] = np
    Uniform(a, b):     E[X] = (a + b) / 2
    Gaussian(μ, σ²):   E[X] = μ
    Categorical(p₁,...,pₖ): E[X] = Σ i × pᵢ   (if outcomes are numbered 1..k)
```

### Why expectation matters in AI

```
Training loss is an expectation:

    The "true" loss is an expectation over ALL possible training examples:
        L = E_{x ~ data} [loss(model, x)]

    We can't compute this (infinite data). So we approximate with a batch:
        L̂ = (1/B) Σ loss(model, xᵢ)     for a batch of B examples

    This batch average is an estimate of the true expectation.
    Larger batches → better estimates (closer to the true E[loss]).

Reward in RLHF:

    The objective is to maximize expected reward:
        J(policy) = E_{response ~ policy} [reward(response)]

    "On average, across all responses the model might generate,
     what reward do we expect?"
```

---

## 2. Variance — How Spread Out Are the Values?

Expectation tells you the center. Variance tells you how much values scatter around that center.

```
Var(X) = E[(X - μ)²]     where μ = E[X]

    "Average squared distance from the mean."

Expanded form:
    Var(X) = E[X²] - (E[X])²

    "Average of squares minus square of average."
    This form is often easier to compute.

Standard deviation:
    σ = √Var(X)

    Same units as X. Variance is in squared units.
    "Most values are within 1-2 standard deviations of the mean."
```

### Concrete example

```
Fair die: E[X] = 3.5

    Var(X) = E[(X - 3.5)²]
           = (1-3.5)²(1/6) + (2-3.5)²(1/6) + (3-3.5)²(1/6)
           + (4-3.5)²(1/6) + (5-3.5)²(1/6) + (6-3.5)²(1/6)
           = (6.25 + 2.25 + 0.25 + 0.25 + 2.25 + 6.25) / 6
           = 17.5 / 6
           ≈ 2.92

    σ = √2.92 ≈ 1.71

    Most rolls land within 3.5 ± 1.71, i.e., roughly 2 to 5. Makes sense.

Constant (always returns 5): Var(X) = 0. No spread.
```

### Key properties

```
Var(aX + b) = a² × Var(X)
    Adding a constant (b) doesn't change spread.
    Scaling by a multiplies variance by a².

Var(X + Y) = Var(X) + Var(Y)     IF X and Y are independent.
    Variances add for independent variables.
    (For dependent variables, there's a covariance term.)

Variance of common distributions:
    Bernoulli(p):      Var = p(1-p)         max at p=0.5
    Binomial(n, p):    Var = np(1-p)
    Uniform(a, b):     Var = (b-a)² / 12
    Gaussian(μ, σ²):   Var = σ²
```

### Why variance matters in AI

**Gradient variance and batch size:**

```
Stochastic gradient (batch size 1):
    gradient = ∇loss(model, x₁)        — one random example
    High variance — the gradient direction is noisy.

Mini-batch gradient (batch size B):
    gradient = (1/B) Σ ∇loss(model, xᵢ)
    Variance of the mean = Var(single) / B

    Batch size 1:    full variance
    Batch size 32:   variance / 32
    Batch size 256:  variance / 256

    Larger batch → lower variance → smoother training → can use larger learning rate.
    But diminishing returns: 32→256 helps a lot, 256→2048 helps less.
```

**Weight initialization variance:**

```
Xavier initialization:
    Var(W) = 2 / (fan_in + fan_out)

Kaiming initialization:
    Var(W) = 2 / fan_in

The variance is chosen so that:
    Var(output) ≈ Var(input)

If variance grows through layers → activations explode.
If variance shrinks → activations vanish.
The right initialization variance keeps signals stable.
```

**Bernoulli variance and dropout:**

```
Dropout mask: each entry ~ Bernoulli(p), where p = keep probability.

    Var(mask) = p(1-p)

    Maximum variance at p = 0.5 (most random).
    At p = 0.9 (standard dropout): Var = 0.09 (moderate noise).
    At p = 1.0 (no dropout): Var = 0 (no randomness).
```

---

## 3. Maximum Likelihood Estimation (MLE)

### The question MLE answers

```
You have some data. You have a family of distributions (parameterized by θ).
Which θ makes the data most likely?

Example: you flip a coin 100 times. 73 heads, 27 tails.
    Model: Bernoulli(θ) where θ = P(heads)

    Which θ makes this data most probable?

    P(data | θ) = θ⁷³ × (1-θ)²⁷

    This is the LIKELIHOOD function — probability of the data given θ.
    MLE finds the θ that maximizes it.

    Answer: θ = 73/100 = 0.73

    Intuitive: the best estimate of the coin's bias is the observed frequency.
```

### The general framework

```
1. Choose a model family:  P(data | θ)
2. Observe data:           x₁, x₂, ..., xₙ
3. Write the likelihood:   L(θ) = P(x₁, x₂, ..., xₙ | θ)
4. Maximize:               θ_MLE = argmax_θ L(θ)

If data points are independent:
    L(θ) = P(x₁|θ) × P(x₂|θ) × ... × P(xₙ|θ)

    Product of probabilities — one per data point.
```

### Log-likelihood (why we use logs)

```
Multiplying many probabilities → tiny numbers → underflow.
Same problem as the chain rule for sentences.

Fix: take the log. Log turns products into sums.

    log L(θ) = log P(x₁|θ) + log P(x₂|θ) + ... + log P(xₙ|θ)

    Maximizing log L(θ) gives the same θ as maximizing L(θ)
    because log is monotonically increasing.

    This sum of log-probabilities is exactly what we optimize.
```

---

## 4. MLE Is What LLM Training Does

### Cross-entropy loss = negative log-likelihood

```
LLM training data: a sequence of tokens x₁, x₂, ..., xₙ.

The model outputs P_model(xₜ | x₁, ..., xₜ₋₁) for each position.

Log-likelihood of the data:
    log L(θ) = Σₜ log P_model(xₜ | x₁, ..., xₜ₋₁)

    "Sum of log-probabilities of each correct next token."

MLE maximizes this. Equivalently, minimize the NEGATIVE log-likelihood:
    NLL = -Σₜ log P_model(xₜ | x₁, ..., xₜ₋₁)

Average over tokens:
    Loss = -(1/T) Σₜ log P_model(xₜ | x₁, ..., xₜ₋₁)

This is the CROSS-ENTROPY LOSS — the standard LLM training loss.
It's not a separate invention. It IS maximum likelihood estimation.
```

### Why minimizing cross-entropy = maximizing likelihood

```
Each training step:
    Input: "The capital of France is"
    True next token: "Paris"

    Model outputs: P("Paris" | context) = 0.82

    Loss for this token = -log(0.82) = 0.198

    If the model had said P("Paris") = 0.01:
        Loss = -log(0.01) = 4.61     much higher loss

    If the model had said P("Paris") = 0.99:
        Loss = -log(0.99) = 0.01     very low loss

    -log(p) is:
        0 when p = 1     (model is certain and correct → no loss)
        ∞ when p → 0     (model is certain something ELSE is correct → infinite loss)

    Minimizing this pushes the model to assign high probability
    to the tokens that actually appear in the training data.
    That's MLE.
```

### The connection chain

```
MLE says:           find θ that maximizes P(data | θ)
For LLMs:           find weights that maximize P(training text | weights)
Decompose:          = Π P(tokenₜ | context, weights)
Take log:           = Σ log P(tokenₜ | context, weights)
Negate and average: = -(1/T) Σ log P(tokenₜ | context, weights)
                    = cross-entropy loss

Every gradient step in LLM training is one step of MLE.
```

---

## 5. MLE for Other Distributions

### Gaussian MLE

```
Data: x₁, x₂, ..., xₙ drawn from N(μ, σ²). Find μ and σ².

    log L(μ, σ²) = Σ log [ (1/σ√(2π)) exp(-(xᵢ-μ)²/2σ²) ]
                 = -(n/2)log(2πσ²) - (1/2σ²) Σ(xᵢ-μ)²

    Take derivative, set to zero:
        μ_MLE = (1/n) Σ xᵢ               the sample mean
        σ²_MLE = (1/n) Σ (xᵢ - μ_MLE)²   the sample variance

    MLE for a Gaussian = just compute mean and variance of your data.
    What you'd do intuitively.
```

### Categorical MLE

```
Data: n tokens from a vocabulary of k types.
    Token "the" appears 500 times, "cat" 30 times, etc.

    P_MLE(token) = count(token) / n

    "The maximum likelihood estimate of a categorical distribution
     is just the frequency of each outcome."

    This is the simplest language model — just count token frequencies.
    A neural LM does the same thing but conditioned on context,
    with a neural network instead of raw counts.
```

---

## 6. Bias-Variance Tradeoff (Brief)

The tradeoff between accuracy and consistency of an estimator.

```
Bias:     does the estimator hit the right answer ON AVERAGE?
Variance: how much does the estimator scatter around its average?

    High bias, low variance:    consistently wrong in the same way.
    Low bias, high variance:    right on average, but noisy.
    Low bias, low variance:     ideal.

              Low variance       High variance
            ┌──────────────┐   ┌──────────────┐
High bias   │  • • •       │   │   •          │
            │  • • •       │   │       •      │
            │  (clustered, │   │ •         •  │
            │   off-center)│   │    (scattered,│
            │              │   │     off-center)│
            └──────────────┘   └──────────────┘
            ┌──────────────┐   ┌──────────────┐
Low bias    │     •••      │   │  •     •     │
            │     •X•      │   │    •X•       │
            │     •••      │   │  •       •   │
            │  (clustered, │   │ (scattered,  │
            │   on-center) │   │  on-center)  │
            └──────────────┘   └──────────────┘
                                X = true value
```

### How this applies to ML

```
Model too simple (high bias):
    Can't capture the true pattern. Underfits.
    Consistently gets it wrong.
    Example: linear regression on curved data.

Model too complex (high variance):
    Fits training data perfectly, including noise. Overfits.
    Different training sets → very different models.
    Example: memorizing training data instead of learning patterns.

LLM scale:
    Larger models → lower bias (can capture more complex patterns)
    But also lower variance (overparameterized models generalize well,
    contradicting the classical tradeoff — this is the "double descent" phenomenon)

    In practice for LLMs: bigger is almost always better,
    the classical bias-variance tradeoff applies less at extreme scale.
```

---

## 7. Putting It Together

```
Concept            What it is                              LLM connection
────────────────────────────────────────────────────────────────────────────
Expectation        weighted average of outcomes             loss = E[per-token loss]
Variance           spread around the mean                   gradient noise, init scale
MLE                find params that maximize P(data|params) LLM training IS MLE
Log-likelihood     log of data probability                  cross-entropy loss
NLL                negative log-likelihood                  the loss function
Bias-variance      accuracy vs consistency tradeoff         model size vs overfitting
```

The core insight: LLM training is maximum likelihood estimation. The cross-entropy loss is the negative log-likelihood. Every gradient step asks: "how do I adjust the weights so the model assigns higher probability to the tokens that actually appeared in the training data?" That's MLE, applied to a categorical distribution conditioned on context.

---

**Next:** `04_Log_Probabilities_and_Softmax.md` — why we work in log-space, how softmax creates distributions, and temperature scaling.

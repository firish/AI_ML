# Probability and Distributions

---

## 1. What Is Probability?

A probability is a number between 0 and 1 that measures how likely something is.

```
P(event) = 0     → impossible
P(event) = 1     → certain
P(event) = 0.3   → happens 30% of the time
```

That's the intuition. More precisely: if you repeated the experiment infinitely many times, the fraction of times the event occurs converges to the probability.

### The setup

```
Experiment:     something with uncertain outcomes (roll a die, flip a coin, generate a token)
Sample space:   the set of ALL possible outcomes (Ω)
Event:          a subset of outcomes you care about

Examples:
    Flip a coin:
        Sample space: {heads, tails}
        Event "heads": {heads}
        P(heads) = 1/2

    Roll a die:
        Sample space: {1, 2, 3, 4, 5, 6}
        Event "even": {2, 4, 6}
        P(even) = 3/6 = 1/2

    LLM generates next token:
        Sample space: entire vocabulary (e.g., 128,000 tokens)
        Event "the": {the}
        P("the") = whatever the model's softmax outputs
```

### The rules

```
1. All probabilities are between 0 and 1:
       0 ≤ P(A) ≤ 1

2. All outcomes sum to 1:
       P(outcome₁) + P(outcome₂) + ... = 1

       For a die: 1/6 + 1/6 + 1/6 + 1/6 + 1/6 + 1/6 = 1
       For an LLM: P(token₁) + P(token₂) + ... + P(token₁₂₈₀₀₀) = 1

3. Complement:
       P(not A) = 1 - P(A)
       P(not heads) = 1 - 0.5 = 0.5

4. Union (either A or B):
       P(A or B) = P(A) + P(B) - P(A and B)
       Subtract the overlap to avoid double-counting.
```

---

## 2. What Is a Distribution?

A distribution is a function that assigns a probability to every possible outcome. It answers: "for each thing that could happen, how likely is it?"

```
A fair die has a distribution:
    P(1) = 1/6,  P(2) = 1/6,  P(3) = 1/6,
    P(4) = 1/6,  P(5) = 1/6,  P(6) = 1/6

A loaded die has a different distribution:
    P(1) = 1/10, P(2) = 1/10, P(3) = 1/10,
    P(4) = 1/10, P(5) = 1/10, P(6) = 1/2

Both are valid distributions (all ≥ 0, sum to 1).
They describe different experiments with different likelihoods.
```

### Discrete vs continuous

```
Discrete: outcomes are countable (die rolls, tokens, coin flips).
    Distribution assigns a probability to each outcome.
    Sum of all probabilities = 1.

    LLMs are discrete: the vocabulary is a finite set.
    The model outputs a discrete distribution over 128K tokens.

Continuous: outcomes are real numbers (height, temperature, weight).
    Can't assign probability to each individual value
    (infinite values → each would be 0).
    Instead, use a density function: P(a ≤ X ≤ b) = area under curve from a to b.
    Total area under curve = 1.

For LLMs, almost everything is discrete. Continuous distributions
show up mainly in weight initialization and noise injection.
```

---

## 3. Bernoulli Distribution — One Coin Flip

The simplest distribution. Two outcomes: success (1) or failure (0).

```
X ~ Bernoulli(p)

    P(X = 1) = p         "success"
    P(X = 0) = 1 - p     "failure"

    One parameter: p (probability of success).

Examples:
    Fair coin:     Bernoulli(0.5)    P(heads) = 0.5
    Loaded coin:   Bernoulli(0.8)    P(heads) = 0.8
    Rare event:    Bernoulli(0.01)   P(success) = 1%
```

### Why Bernoulli matters in AI

**Dropout** is Bernoulli sampling:

```
During training, each neuron is independently kept or dropped:

    mask_i ~ Bernoulli(p)     where p = 0.9 (keep probability)

    For each neuron:
        mask = 1 with probability 0.9 → keep it
        mask = 0 with probability 0.1 → zero it out

    output = mask × neuron_output

    Each neuron is an independent coin flip.
    The mask is a vector of Bernoulli samples.
```

---

## 4. Binomial Distribution — Many Coin Flips

If you flip a Bernoulli coin n times, how many successes do you get?

```
X ~ Binomial(n, p)

    n = number of trials
    p = probability of success per trial
    X = total number of successes (0, 1, 2, ..., n)

    P(X = k) = C(n,k) × p^k × (1-p)^(n-k)
                ^^^^^^   ^^^   ^^^^^^^^^^^
                ways to   k     (n-k)
                choose   wins   losses
                which k
                are wins

Example: flip a fair coin 10 times. P(exactly 7 heads)?

    P(X=7) = C(10,7) × 0.5^7 × 0.5^3
           = 120 × 0.0078 × 0.125
           = 0.117     (about 12%)
```

### Why Binomial matters

Dropout on a layer with 768 neurons (p = 0.9):

```
How many neurons survive? X ~ Binomial(768, 0.9)

    Expected survivors: 768 × 0.9 = 691
    Standard deviation: √(768 × 0.9 × 0.1) ≈ 8.3

    Almost always between ~670 and ~710 neurons survive.
    The randomness forces the network not to rely on any single neuron.
```

---

## 5. Uniform Distribution — Everything Equally Likely

```
Discrete uniform: each of n outcomes has probability 1/n.

    Fair die:          P(k) = 1/6 for k = 1,...,6
    Random token:      P(token) = 1/128000 for each token
    Random index:      P(i) = 1/n for i = 1,...,n

Continuous uniform on [a, b]: constant density = 1/(b-a).

    Uniform(0, 1):  any value between 0 and 1 is equally likely.
    Used for: random number generation, random initialization.
```

### Why Uniform matters in AI

**Random initialization:**

```
Some weight initialization schemes start from a uniform distribution:

    Xavier uniform: W ~ Uniform(-√(6/(fan_in + fan_out)), +√(6/(fan_in + fan_out)))

    fan_in = 768, fan_out = 768:
        W ~ Uniform(-0.063, 0.063)
        Each weight is drawn uniformly from this narrow range.
```

**Maximum entropy:** the uniform distribution has the HIGHEST entropy (uncertainty) of any distribution over a finite set. It represents "I have no idea which outcome is more likely."

```
When an LLM is untrained, its output distribution is close to uniform:
    P("the") ≈ P("banana") ≈ P("zzz") ≈ 1/128000

    Training moves it away from uniform → toward peaked distributions
    that assign high probability to correct next tokens.
```

---

## 6. Categorical Distribution — The LLM Distribution

The most important distribution for understanding language models. A generalization of Bernoulli to more than two outcomes.

```
X ~ Categorical(p₁, p₂, ..., pₖ)

    k possible outcomes (categories).
    P(X = i) = pᵢ
    All pᵢ ≥ 0 and p₁ + p₂ + ... + pₖ = 1

    Bernoulli is just Categorical with k=2.

Example: a 3-word vocabulary:
    P("cat") = 0.7
    P("dog") = 0.2
    P("the") = 0.1
    Sum = 1.0 ✓

    This IS the distribution. Sample from it:
        70% of the time you draw "cat"
        20% of the time you draw "dog"
        10% of the time you draw "the"
```

### This is what an LLM outputs

```
At every generation step, the model produces a categorical distribution
over the entire vocabulary:

    Input: "The capital of France is"

    Model output (logits → softmax → probabilities):
        P("Paris")     = 0.82
        P("the")       = 0.03
        P("Lyon")      = 0.02
        P("a")         = 0.01
        P("located")   = 0.01
        ... 127,995 more tokens with tiny probabilities ...
        Sum = 1.0

    This is a Categorical(0.82, 0.03, 0.02, ...) distribution
    over 128,000 tokens.

    "Generating a token" = sampling from this categorical distribution.
    "Greedy decoding" = always picking the highest probability token.
    "Temperature" = reshaping this distribution (covered in a later file).
```

### Why this is the central object

```
Everything in LLM training and inference revolves around this distribution:

    Training loss:    cross-entropy between model's categorical distribution
                      and the true next token

    Generation:       sample from the categorical distribution

    Evaluation:       how peaked is the distribution? (perplexity)

    RLHF:            compare two categorical distributions (KL divergence)

    Softmax:         the function that converts raw numbers (logits)
                     into a valid categorical distribution

Every time you see "the model predicts the next token," it means:
the model outputs a categorical distribution, and we sample or argmax from it.
```

---

## 7. Gaussian (Normal) Distribution — The Bell Curve

The most important continuous distribution. Appears everywhere because of the Central Limit Theorem: averages of many random things tend to be Gaussian.

```
X ~ Normal(μ, σ²)    also written  X ~ N(μ, σ²)

    μ (mu):    mean — the center of the bell curve
    σ² (sigma squared): variance — how spread out it is
    σ (sigma): standard deviation — square root of variance

    Density function: f(x) = (1 / σ√(2π)) × exp(-(x-μ)² / 2σ²)

    You don't need to memorize the formula.
    What matters: the shape and what μ and σ control.
```

### The shape

```
                    μ = 0, σ = 1 (standard normal)

                         ▄▄████▄▄
                      ▄██████████████▄
                   ▄████████████████████▄
                ▄████████████████████████████▄
           ▄▄████████████████████████████████████▄▄
    ───────────────────────────────────────────────────
           -3σ   -2σ   -1σ    μ    +1σ   +2σ   +3σ

    68% of values fall within 1σ of the mean
    95% of values fall within 2σ of the mean
    99.7% of values fall within 3σ of the mean
```

### What μ and σ control

```
μ shifts the curve left or right (where the center is):
    N(0, 1)  → centered at 0
    N(5, 1)  → centered at 5

σ widens or narrows the curve (how spread out):
    N(0, 0.1²) → very narrow, values tightly clustered around 0
    N(0, 1²)   → standard spread
    N(0, 10²)  → very wide, values spread far from 0
```

### Why Gaussian matters in AI

**Weight initialization:**

```
Kaiming normal initialization:
    W ~ N(0, σ²)    where σ = √(2 / fan_in)

    For fan_in = 768:
        W ~ N(0, 0.051²)
        Most weights between -0.15 and 0.15

    Why Gaussian? The CLT ensures that weighted sums of Gaussian inputs
    stay Gaussian. This keeps activations in a stable range through
    many layers. The variance formula (2/fan_in) is chosen so that
    the output variance matches the input variance.
```

**Layer normalization:**

```
LayerNorm normalizes activations to approximately N(0, 1):

    Given activations x = [x₁, x₂, ..., xₙ]:
        μ = mean(x)
        σ = std(x)
        x_normalized = (x - μ) / σ

    After normalization, the activations have mean ≈ 0, std ≈ 1.
    This prevents values from growing or shrinking as they pass through layers.
```

**Gaussian noise in diffusion models:**

```
Diffusion models (Stable Diffusion, DALL-E) work by:
    1. Adding Gaussian noise to images: x_noisy = x + N(0, σ²)
    2. Training a model to predict and remove the noise
    3. At generation: start from pure Gaussian noise, iteratively denoise

    The entire process is built on the Gaussian distribution.
```

---

## 8. Multivariate Gaussian — Gaussians in High Dimensions

When you have a vector of values instead of a single number, each dimension can be Gaussian, and the dimensions can be correlated.

```
X ~ N(μ, Σ)

    μ: mean VECTOR (d-dimensional)
    Σ: covariance MATRIX (d × d)   — generalizes σ² to multiple dimensions

    1D: μ is a number, σ² is a number
    dD: μ is a vector, Σ is a matrix

    Σ encodes:
        Diagonal entries Σᵢᵢ: variance of dimension i (how spread out)
        Off-diagonal Σᵢⱼ:    covariance between dims i and j (correlation)

    If Σ is diagonal (no off-diagonal entries):
        dimensions are independent — each is its own 1D Gaussian.
    If Σ has off-diagonal entries:
        dimensions are correlated — knowing one tells you about another.
```

### Why multivariate Gaussian matters

```
Weight initialization in practice:

    A weight matrix W (768 × 768) is initialized by drawing each entry
    independently from N(0, σ²). This is equivalent to:

        vec(W) ~ N(0, σ²I)

    A 768²-dimensional Gaussian with diagonal covariance (independent entries).
    Each weight is independent. The covariance matrix is σ² × identity.

Covariance in PCA (connects to linear algebra):

    PCA finds the directions of maximum variance in data.
    These directions are the eigenvectors of the covariance matrix Σ.
    If data is multivariate Gaussian, PCA gives the axes of the ellipse.
```

---

## 9. How Distributions Connect to Each Other

```
Bernoulli(p)              two outcomes: 0 or 1
    ↓ repeat n times
Binomial(n, p)            count of successes in n trials
    ↓ n → ∞, p → 0, np = λ
Poisson(λ)                count of rare events (not common in LLMs, but good to know)

Categorical(p₁,...,pₖ)   k outcomes: generalization of Bernoulli
    ↓ repeat n times, count each category
Multinomial(n, p₁,...,pₖ) counts per category in n trials

Uniform(a, b)             "no information" — maximum entropy for bounded values
Gaussian(μ, σ²)           "some information" — maximum entropy given known mean and variance
```

---

## 10. Putting It Together

```
Distribution     What it models                    Where it shows up in AI
────────────────────────────────────────────────────────────────────────────
Bernoulli        yes/no, keep/drop                 dropout masks
Binomial         count of successes                how many neurons survive dropout
Uniform          all outcomes equally likely        weight init, random sampling
Categorical      one of k outcomes                 LLM output, token prediction
Gaussian         continuous, bell-shaped            weight init, LayerNorm, noise
Multivariate     correlated Gaussian vector         data modeling, PCA, diffusion
Gaussian
```

The categorical distribution is the star for LLMs. Every time the model generates text, it's sampling from a categorical distribution. Training the model means shaping that distribution to put high probability on correct next tokens. Everything else — softmax, cross-entropy, temperature, top-k — is machinery for creating, measuring, or reshaping categorical distributions.

---

**Next:** `02_Conditional_Probability_and_Bayes.md` — P(A|B), why language modeling is conditional probability, and Bayes' theorem.

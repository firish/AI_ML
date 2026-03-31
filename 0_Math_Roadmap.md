# Math Roadmap for Understanding LLMs

What you need, in roughly the order you'll encounter it.

Depth guide: **Intuition** = understand what it does and why. **Working** = follow the math step by step. **Deep** = derive it yourself / read research papers.

---

## Phase 1: Linear Algebra (the language everything is written in)

| # | Concept | Depth | Why |
|---|---------|-------|-----|
| 1 | Vectors, dot products, norms | Working | Attention is built entirely on dot products. Similarity = dot product. Every forward pass = dot products. |
| 2 | Matrix multiplication | Working | Every layer, every projection, every forward pass is a matrix multiply |
| 3 | Transpose | Working | Q×Kᵀ in attention, weight tying in decoders, backprop derivations |
| 4 | Projections (rectangular matrices) | Working | W_q, W_k, W_v, FFN layers, patch embeddings, LoRA |
| 5 | Identity, inverse, determinant | Intuition | Reversibility, singularity, rotation vs reflection |
| 6 | Orthogonal matrices and rotation | Intuition | OPQ, RoPE, understanding "safe" transforms that preserve geometry |
| 7 | Eigenvalues and eigenvectors | Intuition | Why RNNs fail, PCA, optimization landscape, matrix "personality" |
| 8 | Covariance matrix | Intuition | Measures correlation structure. Foundation for PCA and OPQ |
| 9 | PCA | Intuition | Dimensionality reduction, finding the true structure of data |
| 10 | SVD | Intuition | LoRA, compression, Procrustes (OPQ), best low-rank approximation |
| 11 | Rank and low-rank approximation | Intuition | Why LoRA works, why fine-tuning is cheap |
| 12 | Tensor operations / einsum | Intuition | Reading transformer code — multi-head attention is tensor reshaping |
| 13 | Broadcasting | Working | Every framework uses it, easy to get wrong silently |

**Status:** Covered in `08_Linear_Algebra_for_ML.md` and `09_Eigenvalues_and_Eigenvectors.md`

---

## Phase 2: Calculus and Optimization (how models learn)

| # | Concept | Depth | Why |
|---|---------|-------|-----|
| 1 | Derivatives and chain rule | Working | Backpropagation IS the chain rule applied recursively |
| 2 | Partial derivatives / gradients | Working | Gradient = direction of steepest ascent, one partial per parameter |
| 3 | Gradient descent (SGD) | Working | The core training loop — update weights opposite to gradient |
| 4 | Learning rate | Working | Too high = diverge, too low = stuck. Schedules = warmup then decay |
| 5 | Loss landscapes, local minima, saddle points | Intuition | Why training gets stuck, why large models train easier (more escape routes in high dimensions) |
| 6 | Adam optimizer | Intuition | Per-parameter adaptive learning rate using running mean + variance of gradients |
| 7 | Hessian / second derivatives | Intuition | Condition number, why some directions converge faster than others |
| 8 | Gradient clipping | Intuition | Cap gradient magnitude to prevent explosions during training |
| 9 | Weight initialization (Xavier/Kaiming) | Intuition | Wrong initial scale → signals vanish or explode on the very first forward pass |

**Status:** Partially covered in `03_Backpropagation_and_Gradient_Descent.md`

---

## Phase 3: Probability and Statistics (what models are actually doing)

| # | Concept | Depth | Why |
|---|---------|-------|-----|
| 1 | Probability distributions (uniform, Gaussian, categorical) | Working | Model outputs a categorical distribution over the vocabulary |
| 2 | Conditional probability | Working | Language modeling IS P(next token \| previous tokens) |
| 3 | Expectation and variance | Working | Average behavior, spread of values, foundation for everything else |
| 4 | Bayes' theorem | Intuition | Priors, posteriors, foundation for RLHF reward models |
| 5 | Maximum likelihood estimation (MLE) | Working | Training = finding parameters that maximize probability of training data |
| 6 | Log probabilities (logits) | Working | Models output log-probs, not probs — addition replaces multiplication, numerically stable |
| 7 | Softmax | Working | Converts logits to probabilities. Temperature scaling. Why subtract max for stability |
| 8 | Sampling strategies (temperature, top-k, top-p) | Working | How you control randomness in generation |
| 9 | Bernoulli / binomial | Intuition | Dropout = each neuron independently kept with probability p |

**Status:** Not yet covered

---

## Phase 4: Information Theory (measuring what models know)

| # | Concept | Depth | Why |
|---|---------|-------|-----|
| 1 | Entropy | Working | Measures uncertainty. High entropy = many plausible next tokens |
| 2 | Cross-entropy | Working | THE loss function for language models. How well predicted distribution matches truth |
| 3 | Perplexity | Working | exp(cross-entropy). The standard LLM evaluation metric. "Model is choosing between N equally likely tokens" |
| 4 | KL divergence | Intuition | How different two distributions are. Core to RLHF, distillation, VAEs |
| 5 | Bits per token / Shannon bound | Intuition | Theoretical minimum compression. Relevant to quantization limits |

**Status:** Not yet covered

---

## Phase 5: Numerical Precision (why 16-bit works and 4-bit doesn't always)

| # | Concept | Depth | Why |
|---|---------|-------|-----|
| 1 | Floating point (FP32, FP16, BF16) | Working | BF16 = same exponent range as FP32, less precision. Why LLMs train in BF16 not FP16 |
| 2 | Quantization math (scale, zero-point, rounding) | Working | INT8/INT4 inference. KV cache compression (QJL, PolarQuant, TurboQuant) |
| 3 | Numerical stability | Intuition | Why softmax subtracts max, why log-probs not probs, why LayerNorm helps |
| 4 | Mixed precision training | Intuition | FP32 master weights, BF16 forward/backward. Saves memory, matches accuracy |

**Status:** Partially covered in quantization notes (`13_Scalar_Quantization.md`, `14_Product_Quantization.md`, paper notes)

---

## Phase 6: Transformer-Specific Math (putting it all together)

| # | Concept | Depth | Why |
|---|---------|-------|-----|
| 1 | Softmax attention and √d_k scaling | Working | Without scaling, dot products grow with dimension, softmax saturates, gradients vanish |
| 2 | Multi-head attention as parallel projections | Working | Each head projects to a different subspace, attends independently, results concatenated |
| 3 | Positional encoding (sinusoidal, RoPE) | Intuition | Sinusoidal = fixed frequencies. RoPE = 2D rotations encoding relative position |
| 4 | Layer normalization | Working | Normalize to mean 0, variance 1 per token. Keeps magnitudes stable through deep networks |
| 5 | Residual connections | Working | output = layer(x) + x. Direct gradient path prevents vanishing. Pre-norm vs post-norm |
| 6 | Feed-forward network (FFN) | Working | Two linear layers + activation. The "memory" — stores factual knowledge |
| 7 | Gating mechanisms (SwiGLU) | Intuition | Element-wise gating lets network learn which FFN dimensions to activate per input |

**Status:** Partially covered in decoder/LLM notes and attention notes

---

## Phase 7: Training at Scale

| # | Concept | Depth | Why |
|---|---------|-------|-----|
| 1 | Mini-batch SGD | Working | Average gradient over a batch. Larger batch = more stable, less noise |
| 2 | Learning rate warmup + cosine decay | Intuition | Start small (don't blow up random weights), ramp up, then slowly decay |
| 3 | Gradient accumulation | Intuition | Simulate larger batch sizes when GPU memory is limited |
| 4 | Data parallelism vs model parallelism vs pipeline parallelism | Intuition | How you split training across multiple GPUs |
| 5 | Scaling laws (Chinchilla) | Intuition | Optimal compute allocation — predicts loss as a function of model size and data |

**Status:** Not yet covered

---

## Phase 8: Alignment and Fine-tuning

| # | Concept | Depth | Why |
|---|---------|-------|-----|
| 1 | Cross-entropy fine-tuning (SFT) | Working | Same loss as pretraining, just on curated data |
| 2 | LoRA math (low-rank update) | Intuition | SVD connection, rank-r weight update, parameter-efficient |
| 3 | RLHF / reward modeling | Intuition | Train reward model on human preferences (Bradley-Terry), optimize policy with KL penalty |
| 4 | DPO (Direct Preference Optimization) | Intuition | Skip reward model, directly optimize from preference pairs with a clever loss |

**Status:** Partially covered in `06_How_Fine_Tuning_Actually_Works.md` and `11_LoRA_QLoRA_For_LLM.md`

---

## What you DON'T need

- **Abstract algebra / group theory** — unless reading theoretical RoPE papers
- **Measure theory** — all the probability you need is discrete (categorical over tokens)
- **Differential equations** — not relevant to standard transformers
- **Fourier transforms** — some positional encoding papers use them, skippable
- **Convolutions** — only for vision transformer heritage, not needed for LLM understanding

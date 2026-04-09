## Mixture of Experts (MoE)

### The Problem

Bigger models are better (scaling laws), but they're also proportionally more expensive to run. A 70B model does 70B operations per token.

```text
What if most of the model's parameters existed but only a FRACTION
activated for each token? You'd get the knowledge capacity of a
large model with the compute cost of a small one.
```

### How MoE Works

Replace the single FFN in each transformer block with **multiple "expert" FFNs** and a **router** that picks which ones to use:

```text
Standard transformer block:
    Attention → FFN → output
    Every token goes through the SAME FFN.
    All parameters used for every token.

MoE transformer block:
    Attention → Router → {Expert₀, Expert₁, Expert₂, ..., Expert₇} → output
    The router picks 2 out of 8 experts for each token.
    Different tokens may use different experts.
    Only 2/8 = 25% of FFN parameters activated per token.
```

**Toy example:**

```text
8 experts, router picks top-2 per token.

Input: "The cat sat on the mat"

    "The"  → router scores: [0.8, 0.1, 0.5, 0.2, 0.1, 0.3, 0.1, 0.2]
              top-2: Expert 0 (0.8), Expert 2 (0.5)
              output = 0.62 × Expert₀("The") + 0.38 × Expert₂("The")

    "cat"  → router scores: [0.2, 0.7, 0.1, 0.6, 0.1, 0.1, 0.3, 0.1]
              top-2: Expert 1 (0.7), Expert 3 (0.6)
              output = 0.54 × Expert₁("cat") + 0.46 × Expert₃("cat")

    "sat"  → router scores: [0.1, 0.1, 0.3, 0.1, 0.8, 0.2, 0.1, 0.6]
              top-2: Expert 4 (0.8), Expert 7 (0.6)
              output = 0.57 × Expert₄("sat") + 0.43 × Expert₇("sat")

Different tokens use different experts!
Nouns might consistently route to experts 1,3.
Verbs might consistently route to experts 4,7.
Each expert can specialise.
```

### The Router

```text
The router is a small linear layer:

    router_scores = softmax(hidden × W_router)

    W_router shape: (d_model × n_experts) = (4096 × 8)
    Output: 8 scores, one per expert.
    Pick top-k (usually k=2).

    The router is LEARNED during training — it discovers which
    expert should handle which type of token.

Load balancing:
    Problem: the router might send all tokens to Expert 0
    (because it learned Expert 0 is "best") → other experts wasted.

    Fix: auxiliary load-balancing loss
    Penalise the model if expert usage is uneven.
    This encourages spreading tokens across all experts.
```

### MoE Dimensions

```text
Mixtral 8x7B (Mistral):
    Total parameters:    46.7B (8 experts × ~5.6B FFN params + shared attention)
    Active parameters:   12.9B (2 experts active per token)
    Performance:         ≈ LLaMA 2 70B quality
    Inference cost:      ≈ 13B model (only 2 experts compute)

    You get 70B-quality output for 13B-level compute cost.

GPT-4 (rumoured):
    Likely a very large MoE model.
    Total params: ~1.8T (rumoured)
    Active params: ~200-300B per token
    This would explain how it's so capable yet runs at
    practical speeds.

DeepSeek V3:
    Total: 671B parameters
    Active: 37B per token
    256 experts, top-8 routing
```

### Dense vs MoE Trade-offs

```text
| Aspect            | Dense (LLaMA)          | MoE (Mixtral)           |
| ----------------- | ---------------------- | ----------------------- |
| Params used/token | 100% (all params)      | 25-35% (top-k experts)  |
| Total params      | Smaller (7B, 70B)      | Larger (47B, 671B)      |
| Inference speed   | Predictable            | Faster per token        |
| Memory            | Just model weights     | ALL experts must be loaded|
|                   |                        | (even though only 2 used)|
| Training          | Straightforward        | Load balancing is tricky |
| Quality           | Strong at its size     | Matches larger dense    |

Key gotcha: MoE needs ALL parameters in memory, even though
only a fraction activate. A 47B MoE model needs as much GPU memory
as a 47B dense model, even though it computes like a 13B model.
```

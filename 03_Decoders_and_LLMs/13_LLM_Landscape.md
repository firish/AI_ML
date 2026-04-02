## The LLM Landscape — Key Models and How They Differ

### Why This File Exists

You now know the full stack: architecture, tokenization, training, alignment, inference, fine-tuning. This file maps those concepts onto the actual models that matter — what each one does differently and why.

---

## The GPT Series (OpenAI)

### GPT-2 (2019) — The Proof of Concept

```text
Parameters:  1.5B (largest variant)
Training:    40B tokens (WebText — outbound links from Reddit, quality filter)
Context:     1,024 tokens
Tokenizer:   BPE, 50,257 vocab
Architecture: standard decoder-only transformer, 48 layers, d=1600

Key contribution:
    Showed that a decoder-only model trained on next-token prediction
    could do tasks it was never explicitly trained for:
    - Translation (without parallel corpora)
    - Summarisation (without labelled summaries)
    - Q&A (without question-answer pairs)

    This was the first clear demonstration of "emergent abilities"
    from scale + next-token prediction.

    OpenAI initially refused to release the full model,
    claiming it was "too dangerous." (They released it months later.)

What it lacked:
    No instruction tuning. No RLHF. It's purely a text completer.
    Ask it a question → it continues the pattern of questions.
```

### GPT-3 (2020) — Scale Changes Everything

```text
Parameters:  175B (100× GPT-2)
Training:    300B tokens (CommonCrawl + books + Wikipedia + code)
Context:     2,048 tokens
Tokenizer:   BPE, 50,257 vocab (same as GPT-2)
Architecture: 96 layers, d=12288, 96 attention heads

Key contribution:
    In-context learning at scale.
    Give the model a few examples in the prompt, and it generalises:

    "Translate English to French:
     sea otter → loutre de mer
     cheese → fromage
     cat →"    → "chat"

    No weight updates. No fine-tuning. Just examples in the prompt.
    This didn't work at GPT-2 scale. At 175B, it suddenly did.

    This is what made GPT-3 revolutionary — not the architecture
    (identical to GPT-2, just bigger), but the BEHAVIOUR that
    emerged from scale.

Cost:    ~$4-12M to train
GPUs:    ~1,000 V100s, ~1 month
Release: API-only (no weights released)
```

### GPT-3.5 / ChatGPT (2022) — The Alignment Breakthrough

```text
Parameters:  ~175B (same architecture as GPT-3, different training)
Training:    GPT-3 base → SFT on instruction data → RLHF

This is GPT-3 + the alignment stack from file 06:
    1. Start with pre-trained GPT-3
    2. SFT: fine-tune on (instruction, response) pairs
    3. RLHF: train reward model on human preferences → PPO

Result:
    Before (GPT-3):  "What is 2+2?" → "What is 2+3? What is 2+4?"
    After (ChatGPT): "What is 2+2?" → "2+2 equals 4."

    Same base model. The alignment training transformed it from
    a text completer into an assistant.

Why it went viral:
    GPT-3 existed for 2 years as an API. Few people cared.
    ChatGPT wrapped it in a chat interface + alignment.
    100M users in 2 months. The fastest-growing product in history.
```

### GPT-4 (2023) — Multimodal + Quality Jump

```text
Parameters:  rumoured ~1.8T (MoE with 8 experts, ~220B active)
Training:    ~13T tokens (rumoured)
Context:     8K → 32K → 128K (progressive releases)
Architecture: MoE (rumoured), otherwise mostly standard transformer

Key changes:
    1. Multimodal: accepts images as input (not just text)
       Uses a vision encoder (similar to CLIP from file 09 in Phase 1)
       to convert images → vectors, then feeds into the LLM.

    2. Massive quality jump on reasoning, coding, exams:
       - Bar exam: GPT-3.5 = 10th percentile, GPT-4 = 90th percentile
       - SAT Math: GPT-3.5 = 70th percentile, GPT-4 = 89th percentile
       - LeetCode: GPT-4 solves medium/hard problems consistently

    3. Better calibration: knows what it doesn't know (somewhat)
    4. Longer, more coherent outputs

Cost:    ~$100M+ to train (rumoured)
GPUs:    ~25,000 A100s (rumoured)
Release: API-only, no architecture details published
```

### GPT-4o (2024) — Native Multimodal

```text
Key change: "o" = omni

    GPT-4 bolted vision onto a text model (encode image → feed to LLM).
    GPT-4o was trained natively on text + images + audio together.

    This means it can:
    - See images, hear audio, read text — all in one forward pass
    - Generate speech directly (not text-to-speech)
    - Understand tone, emotion, context from voice
    - Respond with appropriate vocal inflection

Architecture:
    Single model handles all modalities, rather than separate
    encoder + LLM pipeline. Faster and more coherent.
```

### o1 / o3 Series (2024-2025) — Reasoning Models

```text
Key change: "thinking" before answering.

    Standard LLM: prompt → answer (one forward pass per token)
    o1/o3: prompt → long internal chain-of-thought → answer

    The model generates a hidden "thinking" trace where it:
    - Breaks the problem into steps
    - Considers multiple approaches
    - Checks its own work
    - Backtracks when it finds errors

    This thinking can be hundreds or thousands of tokens long.
    The user only sees the final answer.

Training:
    Reinforcement learning on reasoning tasks.
    Trained to produce reasoning chains that lead to correct answers.
    NOT just SFT on examples of reasoning — the RL step is crucial.

Trade-off:
    Much better at math, coding, science, logic.
    Much slower and more expensive (generates many more tokens).
    Not better (sometimes worse) at simple tasks where thinking is overkill.
```

---

## The LLaMA Series (Meta) — Open Weights

### Why LLaMA Matters

```text
Before LLaMA (pre-2023):
    If you wanted a good LLM, you used OpenAI's API.
    No one outside big labs had access to strong model weights.

LLaMA changed this:
    Meta released the WEIGHTS (not just an API).
    Anyone can download, run, fine-tune, and build on them.
    This created the entire open-source LLM ecosystem.
```

### LLaMA 1 (Feb 2023)

```text
Sizes:   7B, 13B, 33B, 65B
Training: 1-1.4T tokens (Chinchilla-optimal)
Context:  2,048 tokens
Architecture: standard decoder, RoPE, pre-norm (RMSNorm), SwiGLU FFN

Key insight (from the Chinchilla paper):
    LLaMA 65B (1.4T tokens) matched GPT-3 175B (300B tokens).
    Smaller model + more data = same quality at lower inference cost.

    This validated the "train smaller models on more data" approach
    and made strong LLMs accessible on fewer GPUs.

Release: weights released for "research" → leaked immediately
         → ignited the open-source LLM revolution
```

### LLaMA 2 (Jul 2023)

```text
Sizes:    7B, 13B, 70B
Training: 2T tokens (40% more than LLaMA 1)
Context:  4,096 tokens (2× LLaMA 1)
New:      GQA on 70B model (from file 07), commercial license

Chat versions (LLaMA 2 Chat):
    SFT + RLHF applied (like ChatGPT's training, but open-weight).
    First strong open-weight chat model.

    Quality: roughly GPT-3.5 level on most benchmarks.
```

### LLaMA 3 (Apr 2024) and LLaMA 3.1 (Jul 2024)

```text
Sizes:    8B, 70B, 405B
Training: 15T tokens (7× LLaMA 2!)
Context:  8K → 128K (3.1)
Tokenizer: new, 128K vocab (vs 32K in LLaMA 2) — better multilingual
Architecture: GQA on all sizes, RoPE, RMSNorm, SwiGLU

The 405B model:
    Largest open-weight model released.
    Competitive with GPT-4 on many benchmarks.
    MoE was NOT used — it's a dense 405B model.
    Required 16,000 H100s to train.

Key trend: 15T tokens on 70B params → ratio = 214×
    Far beyond Chinchilla-optimal (20×), but it keeps helping
    because data quality improvements compensate.
```

### LLaMA 4 (2025)

```text
Sizes:    Scout (17B active / 109B total), Maverick (17B active / 400B total)
Architecture: MoE — Meta's first use of Mixture of Experts
    Scout:   16 experts, 1 active per token
    Maverick: 128 experts, 1 active per token
Context:  up to 10M tokens (Scout)
Training: natively multimodal (text + images)

Key change: Meta finally adopted MoE.
    This allows much larger total parameter counts
    while keeping inference cost proportional to active params only.
```

---

## Claude (Anthropic) — Safety-First

### The Approach

```text
Anthropic was founded by ex-OpenAI researchers focused on AI safety.
Claude's distinguishing feature is Constitutional AI (from file 06):

    Instead of RLHF with human labellers deciding what's "good":
    1. Write a constitution (set of principles)
    2. Have the AI critique its own outputs against the principles
    3. Have the AI revise its outputs
    4. Train on the self-improved outputs (RLAIF)

    This scales better than human feedback and makes the
    alignment principles explicit and auditable.
```

### Model Progression

```text
Claude 1 (2023):     competitive with GPT-3.5
Claude 2 (2023):     longer context (100K tokens), better reasoning
Claude 3 (2024):     three tiers — Haiku (fast), Sonnet (balanced), Opus (best)
Claude 3.5 (2024):   Sonnet surpassed many Opus benchmarks at lower cost
Claude 4 (2025):     further quality improvements across all tiers

Key differentiators:
    - Long context: early leader in 100K+ token windows
    - Constitutional AI: explicit safety principles
    - Careful deployment: slower release pace, more safety testing
    - Strong at analysis, writing, nuanced reasoning
```

---

## Gemini (Google DeepMind) — Native Multimodal

### The Approach

```text
Google's advantage: they invented the transformer (2017 paper),
have massive data (Search, YouTube, Books), and own TPU hardware.

Gemini is natively multimodal — trained from the start on
text + images + audio + video together (not bolted on after).

This is different from GPT-4 (text model + vision encoder)
or CLIP (separate encoders for each modality).
Gemini processes all modalities in a single architecture.
```

### Model Progression

```text
Gemini 1.0 (Dec 2023): Ultra, Pro, Nano (3 sizes)
Gemini 1.5 (Feb 2024): Pro with 1M token context window
Gemini 2.0 (Dec 2024): Flash (fast), thinking capabilities
Gemini 2.5 (2025):     Pro with built-in reasoning (thinking)

Key differentiators:
    - 1M+ token context window (largest in production)
    - Native multimodal training (video understanding)
    - Trained on TPUs (Google's custom hardware)
    - Integrated into Google products (Search, Workspace, Android)
    - Gemini Nano runs on-device (phones)
```

---

## Mistral / Mixtral (Mistral AI) — Efficient Open Models

### Why Mistral Matters

```text
French startup. Small team. Focused on efficiency.
Their models punch far above their size class.
```

### Key Models

```text
Mistral 7B (Sep 2023):
    Architecture innovations:
    - Sliding window attention (from file 07): 4,096 token window
      instead of full attention, reduces memory from O(n²) to O(n×w)
    - GQA (8 KV heads for 32 query heads)
    - 32K context but efficient due to sliding window

    Result: beat LLaMA 2 13B on most benchmarks.
    A 7B model outperforming a 13B model.

Mixtral 8x7B (Dec 2023):
    Architecture: MoE (from file 07)
    - 8 experts, 2 active per token
    - 46.7B total params, ~12.9B active per token
    - Same inference cost as a ~13B dense model
    - Quality approaching LLaMA 2 70B

    First widely-used open MoE model.
    Proved MoE works for open-weight community.

Mixtral 8x22B (Apr 2024):
    - 8 experts, 2 active per token
    - 176B total params, ~39B active
    - Competitive with LLaMA 3 70B on many tasks

Mistral Large / Medium / Small:
    Closed-weight commercial models.
    Mistral shifted to also offering API-only models
    alongside their open-weight releases.
```

---

## Other Notable Models

### DeepSeek (DeepSeek AI, China)

```text
DeepSeek-V2 (2024):
    - MoE with Multi-head Latent Attention (MLA):
      compresses KV cache by projecting K,V into a
      smaller latent space before caching.
      Drastically reduces memory.
    - 236B total, 21B active

DeepSeek-V3 (Dec 2024):
    - 671B total, 37B active
    - Trained for ~$5.5M (remarkably cheap for its quality)
    - Competitive with GPT-4o and Claude 3.5 Sonnet

DeepSeek-R1 (Jan 2025):
    - Reasoning model (similar to o1)
    - Open-weight: released weights + training details
    - First open reasoning model competitive with o1
    - Trained primarily with RL (not just SFT on reasoning traces)

Why it matters:
    Showed that frontier-quality models can be trained
    at a fraction of the cost, challenging the
    "only big labs can compete" narrative.
```

### Phi Series (Microsoft)

```text
Phi-1 (2023):    1.3B params, trained on "textbook quality" data
Phi-2 (2023):    2.7B params
Phi-3 (2024):    3.8B params
Phi-4 (2024):    14B params

Key idea: data quality over quantity.
    Phi-3 (3.8B) matches Mixtral 8x7B (46.7B total) on some benchmarks.
    Achieved by training on carefully curated, high-quality data
    + synthetic data generated by larger models (distillation).

    This is the extreme version of the Chinchilla insight:
    with good enough data, tiny models can be surprisingly capable.
```

### Qwen Series (Alibaba)

```text
Qwen 2.5 (2024): 0.5B to 72B sizes
    Strong multilingual (especially Chinese + English).
    Competitive with LLaMA 3 at equivalent sizes.
    Open-weight with commercial licenses.

Qwen-VL: multimodal variants (text + vision)
QwQ:     reasoning variant (thinking model)
```

### Cohere Command R

```text
Focused on RAG (Retrieval-Augmented Generation):
    Built-in citation generation — the model cites which
    retrieved documents support each claim.
    Designed for enterprise search and knowledge tasks.

128K context window, strong multilingual support.
```

---

## Open vs Closed Models

```text
Closed (API-only):                    Open (weights available):
────────────────────                  ──────────────────────────
GPT-4, GPT-4o, o1/o3                 LLaMA 3.1 405B
Claude 3.5/4                         Mistral/Mixtral
Gemini Ultra/Pro                     DeepSeek V3, R1
                                     Qwen 2.5
                                     Phi-4

Advantages of closed:                Advantages of open:
  Higher quality (usually)             Run locally (privacy)
  No infrastructure needed             Fine-tune on your data
  Continuously updated                 No per-token API cost at scale
  Guardrails built in                  Modify architecture
                                       No vendor lock-in
                                       Community innovation

Trade-off:
  Closed: pay per token, trust the provider, limited customisation
  Open:   pay for GPUs, manage infrastructure, full control

The gap is closing:
  2023: GPT-4 >> everything open
  2024: LLaMA 3 405B ≈ GPT-4 on many tasks
  2025: DeepSeek-R1 ≈ o1 on reasoning, open-weight
```

---

## Size Classes — What Runs Where

```text
| Size class    | Examples              | Hardware needed        | Use case              |
| ------------- | --------------------- | ---------------------- | --------------------- |
| ~1-3B         | Phi-3, Gemma 2B       | Phone, laptop CPU      | On-device, edge       |
| ~7-8B         | LLaMA 3 8B, Mistral 7B| 1 consumer GPU (16GB) | Local development     |
| ~13-14B       | Phi-4, Qwen 14B       | 1 GPU (24GB)           | Capable local model   |
| ~30-40B       | Mixtral active params  | 1 A100 (80GB)          | Strong local/cloud    |
| ~65-70B       | LLaMA 3 70B           | 2 A100s or quantized   | Production quality    |
| ~400B+        | LLaMA 3.1 405B        | 8+ A100s/H100s         | Frontier open model   |
| ~1T+ (MoE)    | GPT-4 (rumoured)      | Large cluster           | Frontier closed model |

With quantization (from file 08):
    70B at Q4 ≈ 35GB → fits on 1× A100 or 2× RTX 4090
    8B at Q4 ≈ 4GB → fits on a laptop GPU or even CPU
    3B at Q4 ≈ 1.5GB → runs on a phone
```

---

## Multimodal LLMs — Connecting Back to Phase 1

```text
Remember CLIP, BLIP-2, and vision encoders from Phase 1?
This is where they connect to decoders:

Architecture pattern for vision-language models:
    ┌──────────────┐     ┌───────────────┐     ┌──────────────┐
    │ Image Encoder │ ──→ │  Bridge/      │ ──→ │     LLM      │
    │ (ViT, CLIP)  │     │  Projector    │     │  (Decoder)   │
    └──────────────┘     └───────────────┘     └──────────────┘

    1. Encode image into vectors (Phase 1 encoder)
    2. Project image vectors into the LLM's embedding space
    3. Feed projected vectors as "visual tokens" into the LLM
    4. LLM generates text conditioned on both text and image tokens

Variants:

    LLaVA (2023):
        Vision encoder: CLIP ViT-L
        Bridge: simple linear projection (1 layer!)
        LLM: LLaMA / Vicuna
        Training: 2-stage — align projector, then instruction-tune
        → Surprisingly good for such a simple bridge

    BLIP-2 (2023):
        Vision encoder: ViT-G (EVA-CLIP)
        Bridge: Q-Former (learned queries attend to image features)
        LLM: FlanT5 or OPT
        → More sophisticated bridge, better at details

    GPT-4V / GPT-4o:
        Closed architecture, but likely similar pattern
        with a vision encoder feeding into the LLM.

    Gemini:
        Natively multimodal — NO separate bridge.
        Trained on interleaved text + image tokens from the start.
        The model's own attention layers learn to process both.

The Phase 1 → Phase 3 connection:
    Encoder (Phase 1): turns images/audio/video into vectors
    Decoder (Phase 3): generates text from vectors
    Multimodal LLM: encoder feeds vectors INTO decoder

    Everything we covered connects here.
```

---

## The Architecture Innovations That Matter

```text
Most models use the SAME base architecture (decoder-only transformer).
They differ in which optimisations they adopt:

Innovation            | Introduced by      | Now standard in
──────────────────────|───────────────────|─────────────────
RoPE                  | RoFormer (2021)    | LLaMA, Mistral, most open models
GQA                   | Google (2023)      | LLaMA 2+, Mistral, Gemini
SwiGLU activation     | Google (2020)      | LLaMA, Mistral, most modern LLMs
RMSNorm (pre-norm)    | Zhang & Sennrich   | LLaMA, Mistral (replaces LayerNorm)
MoE routing           | GShard (2020)      | Mixtral, GPT-4(?), LLaMA 4, DeepSeek
Flash Attention       | Tri Dao (2022)     | Everything (training + inference)
Sliding window attn   | Longformer (2020)  | Mistral
Multi-head Latent Attn| DeepSeek (2024)    | DeepSeek V2/V3

The base transformer from 2017 is still there.
These are optimisations bolted on top, not replacements.
```

---

## How to Choose a Model

```text
"I need the best quality, cost doesn't matter"
    → GPT-4o, Claude Opus/Sonnet, Gemini Pro

"I need strong reasoning (math, code, logic)"
    → o3, DeepSeek-R1, Gemini 2.5 Pro (thinking)

"I need to run locally / privately"
    → LLaMA 3.1 70B (quantized) or Mistral/Mixtral

"I need something tiny for a phone/edge device"
    → Phi-3 (3.8B), Gemma 2B, LLaMA 3.2 1B

"I need to fine-tune on my domain data"
    → LLaMA 3.1 8B or 70B + QLoRA (from file 09)

"I need multimodal (images + text)"
    → GPT-4o, Claude Sonnet/Opus, Gemini, LLaVA (open)

"I need the cheapest API for simple tasks"
    → Claude Haiku, GPT-4o mini, Gemini Flash

"I need long context (100K+ tokens)"
    → Gemini 1.5/2.0 (1M), Claude (200K), GPT-4o (128K)
```

---

## The Big Picture Timeline

```text
2017    Transformer paper ("Attention Is All You Need")
2018    GPT-1 (117M) — decoder-only works for language
2018    BERT (340M) — encoder-only dominates NLU benchmarks
2019    GPT-2 (1.5B) — emergent zero-shot abilities
2020    GPT-3 (175B) — in-context learning, few-shot prompting
2021    Codex — GPT-3 fine-tuned on code → GitHub Copilot
2022    ChatGPT — GPT-3.5 + RLHF → 100M users in 2 months
2022    Chinchilla paper — smaller models, more data
2023    GPT-4 — multimodal, massive quality jump
2023    LLaMA 1 — open weights ignite the ecosystem
2023    Mistral 7B — efficiency matters, beats 13B models
2023    Mixtral 8x7B — open MoE works
2024    LLaMA 3 — 15T tokens, 405B open model ≈ GPT-4
2024    Claude 3.5 Sonnet — strong reasoning, long context
2024    Gemini 1.5 — 1M token context window
2024    GPT-4o — native multimodal (text + image + audio)
2024    DeepSeek V3 — frontier quality at $5.5M training cost
2025    o3, DeepSeek-R1 — reasoning models (thinking before answering)
2025    LLaMA 4 — Meta adopts MoE
2025    Gemini 2.5, Claude 4 — continued quality improvements

The trend:
    Phase 1 (2018-2022): make models bigger
    Phase 2 (2022-2024): make training more efficient (data > params)
    Phase 3 (2024-2025): make inference smarter (reasoning, MoE, quantization)
```

---

## Summary

```text
1. GPT series: pioneered scaling, in-context learning, RLHF alignment.
   GPT-4 set the frontier. o1/o3 added explicit reasoning.

2. LLaMA: open weights created the ecosystem. Chinchilla-optimal training.
   Smaller + more data = GPT-3 quality at 1/3 size.

3. Claude: Constitutional AI for alignment. Long context leader.
   Safety-focused approach to deployment.

4. Gemini: native multimodal (not bolted on). 1M token context.
   Google's transformer + data + TPU advantage.

5. Mistral/Mixtral: efficiency innovations (sliding window, MoE).
   Small models beating much larger ones.

6. DeepSeek: frontier quality at fraction of cost. Open reasoning model.
   Challenged the "only big labs can compete" assumption.

7. The base architecture hasn't changed much since 2017.
   Innovation is in: training data, alignment, efficiency tricks,
   and reasoning (thinking before answering).

8. Open vs closed gap is narrowing. The ecosystem is moving toward:
   - Smaller, better-trained models (Phi, Mistral)
   - MoE for scaling without inference cost (Mixtral, LLaMA 4)
   - Reasoning chains for hard problems (o3, R1)
   - Multimodal as default (everything handles text + images)
```

## RAG vs Fine-tuning vs Prompting — When to Use Which

---

## 0. The Central Question

```text
You want an LLM to perform a specific task well in your domain.
You have three primary levers:

    Prompting    →  change what you say to the model, no training
    RAG          →  give the model relevant documents at query time
    Fine-tuning  →  change the model's weights through additional training

These are not competing alternatives — they solve different problems.
Choosing the wrong one wastes money and doesn't fix the actual issue.
```

---

## 1. What Each Approach Actually Does

Understanding what changes in each approach is the foundation for choosing correctly.

### Prompting

```text
What changes: the input to the model.
What stays the same: the model weights (frozen).

The model's knowledge, capabilities, and behaviour are fixed.
Prompting shapes how the model uses what it already knows.

    System prompt: "You are a helpful customer support agent for Acme Corp.
                    Always be concise. Never discuss competitors."
    → The model follows these instructions using its existing capabilities.

    Few-shot examples: "Q: How do I reset my password? A: Click 'Forgot Password'..."
    → The model infers the expected format and style from the examples.

What prompting can do:
    ✓ Set the model's persona, tone, format
    ✓ Provide static context that fits in the window (a few pages of docs)
    ✓ Give instructions for edge cases ("if the user asks X, do Y")
    ✓ In-context learning from a few examples

What prompting cannot do:
    ✗ Add new factual knowledge that wasn't in training data
    ✗ Scale to large, dynamic knowledge bases
    ✗ Teach genuinely new skills the model has never seen
```

### RAG

```text
What changes: the information given to the model at query time.
What stays the same: the model weights (frozen).

    At each query: retrieve relevant chunks → inject into prompt → generate.

The model's underlying capabilities are the same.
RAG changes WHAT the model reads before generating.

What RAG can do:
    ✓ Provide access to large, dynamic, private knowledge bases
    ✓ Ground answers in specific, verifiable documents
    ✓ Keep information fresh (update the index, not the model)
    ✓ Attribute answers to sources
    ✓ Handle long-tail factual queries

What RAG cannot do:
    ✗ Change how the model reasons or behaves
    ✗ Teach new task formats or output structures
    ✗ Improve performance on tasks that require domain-specific reasoning patterns
       (e.g., medical diagnosis reasoning, legal argument structure)
    ✗ Reduce latency — adds retrieval overhead per query
```

### Fine-tuning

```text
What changes: the model's weights via additional training.
What stays the same: the base architecture.

    Training examples: (input, desired output) pairs
    → gradient descent updates the model's parameters
    → the model "learns" from these examples

Full fine-tuning: update all weights. Expensive, requires many GPUs.
LoRA / QLoRA:    update a small low-rank adapter. Much cheaper. (Phase 3, file 12)

What fine-tuning can do:
    ✓ Change the model's default style, tone, format
    ✓ Teach new task structures (e.g., output JSON in a specific schema)
    ✓ Improve performance on domain-specific tasks with distinct patterns
       (medical note summarisation, legal clause extraction)
    ✓ Reduce prompt length (bake instructions into weights instead of system prompt)
    ✓ Distil a large model's behaviour into a smaller, cheaper model

What fine-tuning cannot do:
    ✗ Reliably inject specific factual knowledge
       (facts learned during fine-tuning degrade — the model "forgets" them over time)
    ✗ Keep knowledge current (requires retraining to update)
    ✗ Replace RAG for large knowledge bases
```

---

## 2. The Factual Knowledge Misconception

The most common mistake: using fine-tuning to teach the model facts.

```text
Intuition: "I'll fine-tune the model on our documentation. 
            Then it will know our product and answer correctly."

Why this fails:

    1. Catastrophic forgetting:
       Fine-tuning on new data causes the model to partially forget
       its pre-trained knowledge. Performance on general tasks degrades.

    2. Facts don't stick reliably:
       LLMs don't store facts like a database. They learn statistical patterns.
       A fact mentioned 10 times in fine-tuning data may be learned.
       A fact mentioned once may not be. There's no guarantee.

    3. Hallucination doesn't reduce:
       A fine-tuned model still hallucinate about things NOT in the fine-tuning data.
       And it may confabulate fine-tuned facts with pre-training knowledge in wrong ways.

    4. Knowledge goes stale:
       Product updates? New policies? Fine-tuned knowledge is now wrong.
       Re-fine-tuning costs time and money every time facts change.

Empirical result (from multiple studies):
    Fine-tuned GPT-3.5 on a 100-page knowledge base
    vs RAG with GPT-3.5 + that same knowledge base:
    → RAG outperforms on factual accuracy by 20–40% on average.
    → RAG is cheaper to update (re-index, not retrain).

Rule: Use RAG for factual grounding. Use fine-tuning for behaviour/style.
```

---

## 3. Decision Framework

### The four questions

```text
Ask these four questions in order:

Q1: Does the task require knowledge that isn't in the model's training data?
    (private data, recent events, large knowledge bases, specific documents)
    Yes → RAG is necessary.
    No  → continue to Q2.

Q2: Does the base model already behave the way you want?
    (correct tone, format, task structure, output schema)
    Yes → prompting alone may be sufficient.
    No  → continue to Q3.

Q3: Can you fix the behaviour with a longer / better prompt and examples?
    (few-shot examples, explicit instructions, format demonstrations)
    Yes → improve the prompt. It's cheaper than fine-tuning.
    No  → continue to Q4.

Q4: Is the task a consistent, high-volume pattern where fine-tuning ROI is justified?
    (thousands of queries/day, stable task definition, budget for training)
    Yes → fine-tuning is warranted.
    No  → keep iterating on prompting.
```

### Decision matrix

```text
Situation                                   Best approach
────────────────────────────────────────────────────────────────────────────────
Answer questions from private/internal docs RAG
Knowledge base > a few hundred pages        RAG
Knowledge changes frequently                RAG (update index, not model)
Need source citations and traceability      RAG
Answer is factual, lives in a document      RAG
General task, well-covered in training      Prompting
Task format fits within a system prompt     Prompting
Behaviour change (tone, persona, format)    Prompting first, fine-tuning if needed
Small, consistent style change              Prompting (few-shot examples)
Large, consistent style change at scale     Fine-tuning
Specific output schema (structured JSON)    Prompting first, fine-tuning if needed
Domain reasoning patterns (medical, legal)  Fine-tuning + RAG
High query volume, prompt is very long      Fine-tuning to compress prompt into weights
Reduce cost by using a smaller model        Fine-tuning (distillation from a large model)
```

---

## 4. Cost and Complexity Comparison

```text
Approach    Build cost    Update cost    Per-query cost    Complexity
──────────────────────────────────────────────────────────────────────────────
Prompting   ~hours        ~minutes       Base model cost   Low
RAG         Days–weeks    Hours (re-index) Base + retrieval Medium
Fine-tuning Weeks–months  Weeks           Base model cost  High
                          (retrain)

Fine-tuning (LoRA on 7B model, ~1K examples):
    Training: ~$5-50 on a cloud GPU for a few hours
    High quality training data: much more expensive to curate

Fine-tuning (full fine-tune of 70B model, 100K examples):
    Training: ~$500–$5,000+
    Data curation: often the dominant cost

RAG at scale (1M document chunks):
    Embedding: one-time cost ~$20 with text-embedding-3-small
    Vector DB hosting: ~$70-200/month (managed Pinecone) or self-hosted
    Extra latency per query: +50-200ms for retrieval

Rule of thumb:
    If you're not sure whether you need fine-tuning, you don't need it yet.
    Start with prompting + RAG. Fine-tune only when you have data proving
    the baseline is insufficient and a clear, stable task definition.
```

---

## 5. Combining All Three

These approaches are not mutually exclusive. Production systems often use all three.

### Layered architecture

```text
Layer 1 — Fine-tuning (done once, changes slowly):
    Fine-tune the base model on domain-specific examples.
    Goal: teach the model the right reasoning patterns, output format,
          and communication style for your domain.

    Example: a medical assistant model fine-tuned on clinical note formats
             and medical reasoning patterns.

Layer 2 — RAG (dynamic, per query):
    Retrieve relevant documents at query time.
    Goal: ground the fine-tuned model in specific, current factual information.

    Example: retrieve relevant clinical guidelines, drug interaction data,
             and the specific patient's history for each query.

Layer 3 — Prompting (applied at inference time):
    System prompt with instructions, persona, output constraints.
    Goal: shape the model's behaviour for the specific context.

    Example: "You are a clinical decision support assistant. Always cite
              your sources. Flag urgent findings in [ALERT] tags."

Result:
    The fine-tuned model reasons in the right style for the domain.
    RAG grounds it in current, specific factual information.
    The system prompt shapes output format and safety behaviour.
```

### Real-world example: enterprise customer support bot

```text
Prompting:
    "You are a customer support agent for Acme Corp. Be concise and friendly.
     If you cannot answer, escalate to a human agent."

RAG:
    Index: product documentation, support ticket history, policy documents.
    At each query: retrieve relevant docs → inject into prompt.

Fine-tuning (optional, if needed):
    Fine-tune on (customer query, ideal support response) pairs.
    Goal: match the company's specific support tone and response format.
    When needed: if the base model's tone is wrong and prompting doesn't fix it.

Without fine-tuning (most companies start here):
    Prompting + RAG alone handles 80-90% of cases well.

With fine-tuning added later:
    Higher consistency in tone, shorter prompts (lower cost), better format adherence.
    Justified once query volume is high and tone requirements are proven.
```

---

## 6. Fine-tuning WITH RAG — the Right Way

```text
Common mistake: fine-tune on (question, answer) pairs where the answer
                is factual content from a document.

    Training example:
        Q: "What is the return window for international orders?"
        A: "International orders have a 14-day return window."
    → Teaches the model a fact → fact will degrade, may conflict with RAG context.

Right approach: fine-tune on (question + context, answer) pairs.

    Training example:
        Q: "What is the return window for international orders?"
        Context: "For international customers, returns must be initiated
                  within 14 days of delivery..."
        A: "Based on the policy, international orders have a 14-day return window."

    The model learns:
        → How to read and extract from context (a skill, not a fact)
        → The right tone and format for answers
        → How to cite sources
        → How to say "I don't have that information" when context is absent

    The factual knowledge stays in the RAG index, not the model weights.
    The model learns to USE context well, not to memorise it.
```

---

## 7. When Each Approach Fails

```text
Prompting fails when:
    - Knowledge base is too large for the context window
    - Few-shot examples aren't enough to teach the pattern
    - The model's default behaviour is too far from what you need
    - You need consistent behaviour at scale without relying on long prompts

RAG fails when:
    - The answer doesn't exist in any document (the model needs to reason from first principles)
    - Latency requirements are strict (<100ms — retrieval adds overhead)
    - The query is highly conversational and context-free retrieval works poorly
    - Document quality is poor (garbage in, garbage out)
    - The model lacks domain reasoning skills to interpret retrieved content

Fine-tuning fails when:
    - Training data is insufficient (<100 high-quality examples → overfitting)
    - Task definition is unstable (requirements keep changing)
    - You're trying to teach facts (use RAG instead)
    - The base model is too small for the task complexity
      (fine-tuning a 7B model on medical reasoning won't match a base 70B model)
```

---

## 8. The Practical Starting Point

```text
For any new AI application, this is the right order of operations:

Step 1: Prompting only (1 day)
    Write a good system prompt. Test with 20 real queries.
    Is the output quality acceptable? → ship it.
    Not acceptable? → identify WHY.

Step 2: Add RAG if the problem is factual knowledge (1-2 weeks)
    Does the model not know your domain's facts?
    → Build a RAG pipeline. Chunk, embed, index your documents.
    → Re-evaluate. Is quality acceptable? → ship it.

Step 3: Improve the RAG pipeline before fine-tuning (ongoing)
    Low context recall?  → fix chunking, add hybrid search
    Low faithfulness?    → improve grounding instruction
    Low precision?       → add re-ranking
    Most quality issues are RAG pipeline issues, not model issues.

Step 4: Fine-tune only if there's a clear behavioural gap (weeks)
    Is the model's tone, format, or reasoning style consistently wrong
    in a way that prompting cannot fix?
    Do you have 500+ high-quality (input, ideal output) pairs?
    Is the task definition stable and high-volume?
    → Only then invest in fine-tuning.

This order avoids the common mistake of jumping to fine-tuning
(expensive, slow) when prompting + RAG (cheap, fast to iterate) suffices.
```

---

## Summary

```text
1. Three levers, three different problems:
   Prompting   → shape behaviour using what the model already knows
   RAG         → give the model access to specific, current, large knowledge
   Fine-tuning → change how the model reasons and formats outputs

2. The #1 mistake: using fine-tuning to teach facts.
   Facts belong in the RAG index. Fine-tuning teaches skills and style.

3. Decision order:
   Need factual knowledge from docs? → RAG.
   Behaviour still wrong after good prompting? → fine-tuning.
   Start with prompting + RAG. Fine-tune only when you have evidence it's needed.

4. They combine: fine-tune for reasoning/style, RAG for facts, prompting for constraints.
   Most production systems use all three at different layers.

5. Fine-tuning WITH RAG: train on (question + context, answer) pairs — not (question, answer).
   The model learns to use context, not memorise facts.

6. ROI order: prompting (hours) < RAG (days-weeks) < fine-tuning (weeks-months).
   Always exhaust cheaper options first.
```

## Evaluating RAG — Recall, Faithfulness, and RAGAS

---

## 0. Why RAG Evaluation Is Hard

```text
Evaluating a RAG system is harder than evaluating a classifier.

A classifier: output is a label. Compare to ground truth. Done.
    Accuracy = correct predictions / total predictions

A RAG system: output is a paragraph of text.
    How do you automatically check if it's correct?
    How do you know if the right chunks were retrieved?
    How do you detect subtle hallucinations mixed into a mostly-correct answer?

And the system has multiple components that can fail independently:
    Retrieval can fail   → correct answer never reached the LLM
    Generation can fail  → LLM had the right context but hallucinated anyway
    Both can fail        → wrong chunks + wrong answer

You need metrics that tell you WHICH component is failing,
not just "the system gave a bad answer."
```

---

## 1. The Two Axes of Evaluation

Before diving into metrics, understand the two dimensions you're measuring:

```text
Axis 1: Retrieval quality
    Did we find the right chunks?
    Were the chunks we retrieved actually relevant?
    Did we miss any important chunks?

Axis 2: Generation quality
    Did the LLM use the retrieved context correctly?
    Did it hallucinate anything not in the context?
    Did it actually answer the question asked?

A system can fail on either axis independently:

    Good retrieval, bad generation:
        Retrieved the correct policy document.
        LLM ignored it and made up an answer.
        → Retrieval metrics: high. Generation metrics: low.

    Bad retrieval, good generation:
        Retrieved an irrelevant chunk about shipping.
        LLM correctly summarised the irrelevant chunk.
        → Retrieval metrics: low. Generation metrics: high (given what it had).
        → End-to-end: wrong answer.

Measuring both axes separately tells you where to fix the system.
```

---

## 2. The RAGAS Framework

RAGAS (Retrieval Augmented Generation Assessment) is the standard evaluation
framework for RAG. It defines four core metrics that cover both axes.

```text
RAGAS metrics:

    ┌──────────────────────┬────────────────────┬───────────────────────────────┐
    │ Metric               │ Axis               │ Question it answers           │
    ├──────────────────────┼────────────────────┼───────────────────────────────┤
    │ Faithfulness         │ Generation         │ Is the answer grounded in     │
    │                      │                    │ the retrieved context?        │
    ├──────────────────────┼────────────────────┼───────────────────────────────┤
    │ Answer Relevance     │ Generation         │ Does the answer address       │
    │                      │                    │ the question asked?           │
    ├──────────────────────┼────────────────────┼───────────────────────────────┤
    │ Context Precision    │ Retrieval          │ Are the retrieved chunks      │
    │                      │                    │ actually relevant?            │
    ├──────────────────────┼────────────────────┼───────────────────────────────┤
    │ Context Recall       │ Retrieval          │ Did we retrieve all the       │
    │                      │                    │ chunks needed to answer?      │
    └──────────────────────┴────────────────────┴───────────────────────────────┘
```

Each metric has a specific computation method. Let's go through all four.

---

## 3. Faithfulness

**Question:** Does every claim in the generated answer appear in the retrieved context?

### Computation

```text
Input:
    Answer:  "International returns must be initiated within 14 days [2].
              Customs duties are non-refundable [2]. You can return items
              for up to 30 days if purchased in-store [?]."

    Context: [1] "Refunds accepted within 30 days for standard purchases."
             [2] "International returns: 14-day window. Customs non-refundable."

Step 1: Decompose the answer into atomic claims.
    Claim A: "International returns must be initiated within 14 days."
    Claim B: "Customs duties are non-refundable."
    Claim C: "You can return items for up to 30 days if purchased in-store."

Step 2: For each claim, check: is this claim supported by any retrieved chunk?
    Claim A → supported by chunk [2]                   → 1
    Claim B → supported by chunk [2]                   → 1
    Claim C → NOT in any retrieved chunk               → 0  ← hallucination!

Step 3: Faithfulness = supported claims / total claims
    = 2 / 3 = 0.67

Perfect faithfulness = 1.0 (every claim is grounded in context)
```

### What low faithfulness means

```text
Low faithfulness = the LLM is adding information not present in the retrieved context.
This is the hallucination signal. The answer may sound correct but contains
fabricated facts.

Fix: stronger grounding instruction, lower temperature, better context formatting.
```

---

## 4. Answer Relevance

**Question:** Does the generated answer actually address what was asked?

### Computation

```text
Input:
    Question: "What is the refund policy for international orders?"
    Answer:   "Our company was founded in 2015. We operate in 50 countries
               and offer a wide range of products. Customer satisfaction is
               our top priority."

This answer is on-topic only in the loosest sense. It doesn't answer the question.

RAGAS computation:
    Step 1: Use an LLM to generate n (e.g. 3-5) questions that the answer would
            correctly respond to.

        Generated Q1: "Tell me about your company history."
        Generated Q2: "How many countries do you operate in?"
        Generated Q3: "What is your company's core value?"

    Step 2: Embed the original question and each generated question.

    Step 3: Answer Relevance = mean cosine similarity between
            original question and generated questions.

        sim(original, Q1) = 0.21   ← very different
        sim(original, Q2) = 0.19
        sim(original, Q3) = 0.18

        Answer Relevance = mean(0.21, 0.19, 0.18) = 0.19   ← low. Answer is off-topic.

If the answer truly addressed "refund policy for international orders",
the LLM would generate questions like "What are the international return rules?"
which would have high cosine similarity to the original.

Perfect answer relevance ≈ 1.0 (generated questions closely match original question)
```

### What low answer relevance means

```text
Low answer relevance = the answer is not addressing the question.
Causes:
    - LLM went off-topic
    - Retrieved context was about a related-but-different topic
    - Question was ambiguous and LLM answered a different interpretation

Fix: better retrieval precision (retrieve more on-topic chunks),
     clearer question formulation, query rewriting.
```

---

## 5. Context Precision

**Question:** Among all the chunks retrieved, how many were actually useful?

### Computation

```text
Retrieved chunks (k=4):
    Chunk 1: "International returns: 14-day window."        ← relevant ✓
    Chunk 2: "Shipping times: 7-14 days internationally."  ← not relevant ✗
    Chunk 3: "Customs duties are the customer's cost."     ← relevant ✓
    Chunk 4: "Expedited shipping not available in all."    ← not relevant ✗

Ground truth: chunks 1 and 3 are relevant to "refund policy for international orders."

Context Precision = fraction of retrieved chunks that are relevant, weighted by rank.

Standard Precision@k (simplified version):
    Relevant chunks in top-k / k
    = 2 / 4 = 0.50

Rank-weighted version (Average Precision):
    Rewards having relevant chunks near the top of the retrieved list.

    Precision@1: chunk 1 relevant → 1/1 = 1.0
    Precision@2: chunks 1,2, only 1 relevant → 1/2 = 0.50
    Precision@3: chunks 1,2,3, two relevant → 2/3 = 0.67
    Precision@4: 2/4 = 0.50

    Average Precision = mean of Precision@k at each relevant position
    = mean(1.0, 0.67) = 0.835  (only computed at positions where a relevant chunk appears)

High context precision = most of what you retrieved was actually useful.
Low context precision = retriever is returning noise alongside signal.
```

### What low context precision means

```text
Low precision = retrieved chunks contain too much irrelevant content.
The LLM's context is polluted. The relevant signal is diluted.

Fix: better embedding model, higher similarity threshold cutoff,
     re-ranking (file 07), smaller k with higher quality chunks.
```

---

## 6. Context Recall

**Question:** Did retrieval find all the information needed to construct a correct answer?

### Computation

```text
This metric requires a ground-truth answer (human-written reference answer).

Ground truth answer:
    "International orders must be returned within 14 days. Customs duties
     are non-refundable. Items must be in original packaging."

Step 1: Decompose ground truth into atomic claims.
    Claim A: "International orders must be returned within 14 days."
    Claim B: "Customs duties are non-refundable."
    Claim C: "Items must be in original packaging."

Step 2: For each claim, check: is there a retrieved chunk that supports it?

    Retrieved chunks:
        Chunk 1: "International returns: 14-day window."
        Chunk 2: "Shipping times: 7-14 days."

    Claim A → Chunk 1 supports it  ✓
    Claim B → NOT in any retrieved chunk  ✗  ← retrieval missed this
    Claim C → NOT in any retrieved chunk  ✗  ← retrieval missed this

Step 3: Context Recall = claims supported by context / total claims
    = 1 / 3 = 0.33   ← retrieval missed 2/3 of what was needed

Perfect context recall = 1.0 (all ground truth claims are present in retrieved chunks)
```

### What low context recall means

```text
Low recall = the retriever failed to find some of the chunks needed to answer correctly.
Even if the LLM generates faithfully, the answer will be incomplete.

Fix: increase k, improve chunking (maybe the relevant content was split across
     chunk boundaries), better embedding model, hybrid search to catch keyword matches.
```

---

## 7. How the Four Metrics Diagnose Failures

```text
┌──────────────────┬────────────────────┬────────────────────────────────────────┐
│ Pattern          │ Likely cause       │ Fix                                    │
├──────────────────┼────────────────────┼────────────────────────────────────────┤
│ Low faithfulness │ LLM hallucinating  │ Stronger grounding instruction,        │
│ High relevance   │ on retrieved ctx   │ lower temperature, better formatting   │
├──────────────────┼────────────────────┼────────────────────────────────────────┤
│ Low relevance    │ Retrieval found    │ Better retrieval, query rewriting,     │
│ High faithfulness│ off-topic chunks;  │ check embedding model quality          │
│                  │ LLM answered them  │                                        │
├──────────────────┼────────────────────┼────────────────────────────────────────┤
│ Low precision    │ Noisy retrieval    │ Re-ranking, higher similarity cutoff,  │
│                  │                    │ reduce k                               │
├──────────────────┼────────────────────┼────────────────────────────────────────┤
│ Low recall       │ Missing chunks     │ Increase k, fix chunking,             │
│                  │                    │ hybrid search, better embeddings       │
├──────────────────┼────────────────────┼────────────────────────────────────────┤
│ Low precision    │ Retriever returns  │ Re-ranking is the primary fix          │
│ High recall      │ correct chunks     │ (retrieve many, re-rank to few)        │
│                  │ + noise            │                                        │
├──────────────────┼────────────────────┼────────────────────────────────────────┤
│ All metrics high │ System working     │ Focus on harder queries, edge cases    │
│ but wrong answer │ but ground truth   │ or ground truth may be wrong           │
│                  │ is incomplete      │                                        │
└──────────────────┴────────────────────┴────────────────────────────────────────┘
```

---

## 8. Building an Evaluation Dataset

You can't run RAGAS without a dataset of (question, ground-truth answer, ground-truth relevant chunks) triples. Building this dataset is the hardest part.

### Manual annotation (gold standard)

```text
Domain experts write:
    - A diverse set of questions (covering different topics, difficulty levels)
    - A ground truth answer for each question
    - Which chunks from the corpus contain the answer

Pros: highest quality, reflects real use cases
Cons: expensive, slow, requires domain expertise

Typical eval set size: 50–500 question-answer pairs.
    50 pairs: fast to create, high variance in scores (noisy signal).
    200+ pairs: more reliable signal, expensive to create.
```

### Synthetic generation (scalable)

```text
Use an LLM to generate questions from your existing chunks.

Step 1: Take a chunk from your corpus.
    "International returns must be initiated within 14 days of delivery.
     Customs duties are the customer's responsibility."

Step 2: Ask an LLM to generate questions that this chunk answers.
    → "What is the return window for international orders?"
    → "Who pays customs duties on returns?"
    → "What happens if I miss the international return deadline?"

Step 3: The chunk itself becomes the ground-truth context for each question.
    Generate a ground-truth answer from the chunk for each question.

Step 4: Add negatives (important!):
    For each question, include chunks that are RELATED BUT NOT RELEVANT
    as part of the evaluation. Tests whether precision metrics work.

Pros: scalable to thousands of examples in hours.
Cons: LLM-generated questions can be too easy (always match the source chunk well).
      Doesn't capture the distribution of real user queries.

Best practice: combine synthetic (for volume) + manual (for realism and hard cases).
```

### RAGAS synthetic generation

```text
The RAGAS library includes a TestsetGenerator that automates this process.

from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context

generator = TestsetGenerator.from_langchain(llm, embeddings)
testset = generator.generate_with_langchain_docs(
    documents,
    test_size=100,
    distributions={
        simple: 0.5,         # straightforward single-chunk questions
        reasoning: 0.25,     # requires inferring or combining info
        multi_context: 0.25  # requires multiple chunks to answer
    }
)
```

---

## 9. LLM-as-Judge

Most RAGAS metrics require an LLM to perform the evaluation steps.
This is called "LLM-as-judge."

```text
For faithfulness:
    LLM decomposes the answer into claims.
    LLM checks each claim against the context.

For answer relevance:
    LLM generates reverse questions from the answer.

For context recall:
    LLM decomposes the ground-truth answer into claims.
    LLM checks each claim against retrieved chunks.

The evaluator LLM is separate from the generation LLM.
Typically a powerful model (GPT-4o, Claude Opus) is used as the judge,
even if a smaller model generated the original answer.

Why this works:
    Checking if a claim is supported by a text is easier than generating the claim.
    Strong models can reliably judge faithfulness and relevance.

Limitations:
    LLM judges have their own biases. They tend to prefer well-formatted,
    verbose answers and may rate confident-sounding hallucinations highly.
    Always validate LLM judge scores against human annotations on a sample.
```

---

## 10. Running RAGAS

```text
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from datasets import Dataset

data = {
    "question":  ["What is the international return window?"],
    "answer":    ["International returns must be initiated within 14 days."],
    "contexts":  [["International returns: 14-day window. Customs non-refundable.",
                   "Shipping times: 7-14 days internationally."]],
    "ground_truth": ["International orders must be returned within 14 days."],
}

dataset = Dataset.from_dict(data)

result = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
)

print(result)
# {
#   'faithfulness':       0.92,
#   'answer_relevancy':   0.88,
#   'context_precision':  0.50,   ← half the retrieved chunks were irrelevant
#   'context_recall':     0.67,   ← missed 1/3 of ground truth claims
# }
```

---

## 11. Beyond RAGAS — Other Evaluation Approaches

### Answer correctness (end-to-end)

```text
Compare the generated answer to a human-written reference answer.

Methods:
    Exact match:     answer == ground_truth (too strict, misses paraphrases)
    ROUGE/BLEU:      n-gram overlap (misses semantic equivalence)
    BERTScore:       embedding similarity between answer and ground truth
    LLM judge:       "Does this answer correctly address the question given
                      this reference answer? Score 1-5."

LLM judge is the most reliable for open-ended answers.
BERTScore is a decent automated metric when LLM calls are too expensive.
```

### Retrieval-only metrics (information retrieval standards)

```text
If you have ground-truth relevant chunk labels, you can use classic IR metrics:

Recall@k:
    Of all relevant chunks that exist, what fraction did top-k retrieval find?
    = |relevant ∩ retrieved| / |relevant|

Precision@k:
    Of the k retrieved chunks, what fraction are relevant?
    = |relevant ∩ retrieved| / k

MRR (Mean Reciprocal Rank):
    1 / rank of first relevant chunk, averaged across queries.
    MRR = 0 means the relevant chunk was never in top-k.
    MRR = 1 means the relevant chunk was always rank 1.

NDCG@k (Normalised Discounted Cumulative Gain):
    Rewards relevant chunks appearing at higher ranks.
    Used in the MTEB benchmark (file 03).
    Most rigorous retrieval metric.
```

### Human evaluation

```text
For high-stakes applications, automated metrics aren't enough.
Have domain experts rate answers on:

    Accuracy:       Is the answer factually correct?         (1-5)
    Completeness:   Does it cover all aspects of the question? (1-5)
    Faithfulness:   Is it grounded in the provided sources?  (1-5)
    Usefulness:     Would this answer satisfy the user?      (1-5)

Human eval is slow and expensive. Use it for:
    - Validating that automated metrics correlate with human judgement
    - Final quality gates before production launch
    - Debugging specific failure modes that automated metrics don't catch
```

---

## 12. Evaluation as a Development Loop

```text
Evaluation is most useful as a feedback loop during development, not just at launch.

┌──────────────────────────────────────────────────────────────┐
│                                                              │
│  Build eval dataset (200 questions)                          │
│         │                                                    │
│         ▼                                                    │
│  Baseline RAG system                                         │
│         │                                                    │
│         ▼                                                    │
│  Run RAGAS → identify lowest metric                          │
│         │                                                    │
│         ├── Low context recall?                              │
│         │       → Try larger k, fix chunking, hybrid search  │
│         │                                                    │
│         ├── Low context precision?                           │
│         │       → Try re-ranking, higher similarity cutoff   │
│         │                                                    │
│         ├── Low faithfulness?                                │
│         │       → Improve grounding instruction, lower temp  │
│         │                                                    │
│         └── Low answer relevance?                            │
│                 → Improve query preprocessing, better embeds │
│         │                                                    │
│         ▼                                                    │
│  Re-run RAGAS → check if metric improved, check for regressions
│         │                                                    │
│         ▼                                                    │
│  Repeat until all metrics satisfactory                       │
│                                                              │
└──────────────────────────────────────────────────────────────┘

Key: changing one component often improves one metric but hurts another.
     Always re-run the full suite after any change.
```

---

## Summary

```text
1. RAG has multiple failure modes — retrieval failures and generation failures.
   You need metrics that diagnose WHICH component is failing.

2. RAGAS four core metrics:
   Faithfulness:      are all claims in the answer supported by the context?   (generation)
   Answer Relevance:  does the answer actually address the question?            (generation)
   Context Precision: are the retrieved chunks relevant?                        (retrieval)
   Context Recall:    did retrieval find all chunks needed for a complete answer?(retrieval)

3. All four metrics are computed by an LLM-as-judge (no human labels needed at eval time).
   Context Recall requires a ground-truth answer to compare against.

4. Eval dataset: 200+ (question, ground truth answer) pairs.
   Build with synthetic generation + manual annotation for hard cases.

5. Diagnostic logic:
   Low precision → noisy retrieval → re-ranking
   Low recall → missing chunks → larger k, better chunking, hybrid search
   Low faithfulness → hallucination → stronger grounding instruction
   Low relevance → off-topic answer → better retrieval or query rewriting

6. Evaluation is a loop, not a one-time gate.
   Run after every change. Watch for regressions.
```
